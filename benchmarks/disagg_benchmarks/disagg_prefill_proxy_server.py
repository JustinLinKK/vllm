# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
from quart import Quart, Response, make_response, request
import zmq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PING_SECONDS = 5


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="vLLM P/D disaggregation proxy server")

    # Add args
    parser.add_argument(
        "--timeout",
        type=float,
        default=6 * 60 * 60,
        help="Timeout for backend service requests in seconds (default: 21600)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--prefill-url",
        type=str,
        default="http://127.0.0.1:8100",
        help="Prefill service base URL (protocol + host[:port])",
    )
    parser.add_argument(
        "--decode-url",
        type=str,
        default="http://127.0.0.1:8200",
        help="Decode service base URL (protocol + host[:port])",
    )
    parser.add_argument(
        "--kv-host",
        type=str,
        default="127.0.0.1",
        help="Hostname or IP used by KV transfer (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--prefill-kv-port",
        type=int,
        default=14579,
        help="Prefill KV port (default: 14579)",
    )
    parser.add_argument(
        "--decode-kv-port",
        type=int,
        default=14580,
        help="Decode KV port (default: 14580)",
    )
    parser.add_argument(
        "--discovery-host",
        type=str,
        default="127.0.0.1",
        help="Host for the ZMQ discovery/router socket (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--discovery-port",
        type=int,
        default=30001,
        help="Port for the ZMQ discovery/router socket (default: 30001)",
    )

    return parser.parse_args()


def _listen_for_register(
    poller: zmq.Poller,
    router_socket: zmq.Socket,
    prefills: dict[str, tuple[str, float]],
    decodes: dict[str, tuple[str, float]],
) -> None:
    while True:
        try:
            socks = dict(poller.poll())
        except zmq.ZMQError:
            return
        if router_socket not in socks:
            continue

        try:
            remote_address, message = router_socket.recv_multipart()
        except zmq.ZMQError:
            return
        data = msgpack.loads(message)
        role = data.get("type")
        http_address = data.get("http_address")
        zmq_address = data.get("zmq_address")
        expiry = time.time() + DEFAULT_PING_SECONDS

        if role == "P":
            prefills[http_address] = (zmq_address, expiry)
        elif role == "D":
            decodes[http_address] = (zmq_address, expiry)
        else:
            logger.warning(
                "Unexpected discovery message from %s: %s",
                remote_address,
                data,
            )


def start_service_discovery(
    hostname: str,
    port: int,
) -> tuple[threading.Thread, zmq.Socket, zmq.Context]:
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Discovery port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    prefills: dict[str, tuple[str, float]] = {}
    decodes: dict[str, tuple[str, float]] = {}
    listener = threading.Thread(
        target=_listen_for_register,
        args=(poller, router_socket, prefills, decodes),
        daemon=True,
    )
    listener.start()
    return listener, router_socket, context


def main():
    """parse command line arguments"""
    args = parse_args()

    # Initialize configuration using command line parameters
    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    PREFILL_SERVICE_URL = args.prefill_url
    DECODE_SERVICE_URL = args.decode_url
    PORT = args.port

    PREFILL_KV_ADDR = f"{args.kv_host}:{args.prefill_kv_port}"
    DECODE_KV_ADDR = f"{args.kv_host}:{args.decode_kv_port}"

    logger.info(
        "Proxy resolved KV addresses -> prefill: %s, decode: %s",
        PREFILL_KV_ADDR,
        DECODE_KV_ADDR,
    )

    discovery_thread, discovery_socket, discovery_context = start_service_discovery(
        args.discovery_host,
        args.discovery_port,
    )

    app = Quart(__name__)

    # Attach the configuration object to the application instance so helper
    # coroutines can read the resolved backend URLs and timeouts without using
    # globals.
    app.config.update(
        {
            "AIOHTTP_TIMEOUT": AIOHTTP_TIMEOUT,
            "PREFILL_SERVICE_URL": PREFILL_SERVICE_URL,
            "DECODE_SERVICE_URL": DECODE_SERVICE_URL,
            "PREFILL_KV_ADDR": PREFILL_KV_ADDR,
            "DECODE_KV_ADDR": DECODE_KV_ADDR,
        }
    )

    def _normalize_base_url(url: str) -> str:
        """Remove any trailing slash so path joins behave predictably."""
        return url.rstrip("/")

    PREFILL_BASE = _normalize_base_url(PREFILL_SERVICE_URL)
    DECODE_BASE = _normalize_base_url(DECODE_SERVICE_URL)

    def _build_headers(request_id: str) -> dict[str, str]:
        """Construct the headers expected by vLLM's P2P disagg connector."""
        headers: dict[str, str] = {"X-Request-Id": request_id}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _run_prefill(
        request_path: str,
        payload: dict,
        headers: dict[str, str],
        request_id: str,
    ) -> dict[str, Any]:
        url = f"{PREFILL_BASE}{request_path}"
        start_ts = time.perf_counter()
        logger.info("[prefill] start request_id=%s url=%s", request_id, url)
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Prefill backend error {resp.status}: {error_text}"
                    )
                response_json = await resp.json()
                logger.info(
                    "[prefill] done request_id=%s status=%s elapsed=%.2fs",
                    request_id,
                    resp.status,
                    time.perf_counter() - start_ts,
                )
                return response_json
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Prefill service timeout at {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Prefill service unavailable at {url}") from exc

    async def _stream_decode(
        request_path: str,
        payload: dict,
        headers: dict[str, str],
        request_id: str,
    ):
        url = f"{DECODE_BASE}{request_path}"
        # Stream tokens from the decode service once the prefill stage has
        # materialized KV caches on the target workers.
        logger.info("[decode] start request_id=%s url=%s", request_id, url)
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "Decode backend error %s - %s", resp.status, error_text
                    )
                    err_msg = (
                        '{"error": "Decode backend error ' + str(resp.status) + '"}'
                    )
                    yield err_msg.encode()
                    return
                logger.info(
                    "[decode] streaming response request_id=%s status=%s",
                    request_id,
                    resp.status,
                )
                async for chunk_bytes in resp.content.iter_chunked(1024):
                    yield chunk_bytes
                logger.info("[decode] finished streaming request_id=%s", request_id)
        except asyncio.TimeoutError:
            logger.error("Decode service timeout at %s", url)
            yield b'{"error": "Decode service timeout"}'
        except aiohttp.ClientError as exc:
            logger.error("Decode service error at %s: %s", url, exc)
            yield b'{"error": "Decode service unavailable"}'

    async def _run_decode_json(
        request_path: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        request_id: str,
    ) -> dict[str, Any]:
        """Fetch a standard non-streaming JSON completion from the decode stage."""
        url = f"{DECODE_BASE}{request_path}"
        logger.info(
            "[decode] request json request_id=%s url=%s",
            request_id,
            url,
        )
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Decode backend error {resp.status}: {error_text}"
                    )
                body = await resp.json()
                logger.info(
                    "[decode] json response request_id=%s status=%s",
                    request_id,
                    resp.status,
                )
                return body
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Decode service timeout at {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Decode service unavailable at {url}") from exc

    async def process_request():
        """Process a single request through prefill and decode stages"""
        try:
            original_request_data = await request.get_json()
            client_wants_stream = bool(original_request_data.get("stream", False))

            # Create prefill request (max_tokens=1)
            prefill_request = original_request_data.copy()
            prefill_request["stream"] = False
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            # Execute prefill stage
            # The request id encodes both KV socket addresses so the backend can
            # shuttle tensors directly via NCCL once the prefill response
            # completes.
            request_id = (
                f"___prefill_addr_{PREFILL_KV_ADDR}___decode_addr_"
                f"{DECODE_KV_ADDR}_{uuid.uuid4().hex}"
            )

            headers = _build_headers(request_id)
            await _run_prefill(request.path, prefill_request, headers, request_id)

            # Pass the unmodified user request so the decode phase can continue
            # sampling with the already-populated KV cache.
            if client_wants_stream:
                generator = _stream_decode(
                    request.path, original_request_data, headers, request_id
                )
                response = await make_response(generator)
                response.timeout = None  # Disable timeout for streaming response
                response.content_type = "text/event-stream"
                return response

            body = await _run_decode_json(
                request.path, original_request_data, headers, request_id
            )
            return Response(
                response=json.dumps(body),
                status=200,
                content_type="application/json",
            )

        except Exception:
            logger.exception("Error processing request")
            return Response(
                response=b'{"error": "Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/healthz", methods=["GET"])
    async def healthz():
        return Response(
            response=b'{"status":"ok"}',
            status=200,
            content_type="application/json",
        )

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/v1/chat/completions", methods=["POST"])
    async def handle_request():
        """Handle incoming API requests with concurrency and rate limiting"""
        try:
            return await process_request()
        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    # Start the Quart server with host can be set to 0.0.0.0
    try:
        app.run(host="127.0.0.1", port=PORT)
    finally:
        discovery_socket.close(linger=0)
        discovery_context.term()
        discovery_thread.join(timeout=1)


if __name__ == "__main__":
    main()
