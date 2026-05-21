# python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py --timeout 300
# sleep 1

echo "Testing the proxy server by sending a request to http://localhost:8000/v1/completions"
output=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/shared/models/hf/Meta-Llama-3-8B-Instruct",
"prompt": "San Francisco is a beautiful city, isn'"'"'t it?",
"max_tokens": 10,
"temperature": 0
}')

echo "Response from the proxy server: ${output}"