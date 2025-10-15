for i in {1..10}; do curl -s http://localhost:11434/api/generate \
  -d '{"model":"gpt-oss:20b","prompt":"warm up","stream":false}' >/dev/null; done
for i in {1..10}; do curl -s http://localhost:11434/api/generate \
  -d '{"model":"yi:34b","prompt":"warm up","stream":false}' >/dev/null; done
for i in {1..10}; do curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","prompt":"warm up","stream":false}' >/dev/null; done
for i in {1..10}; do curl -s http://localhost:11434/api/generate \
  -d '{"model":"mistral:7b","prompt":"warm up","stream":false}' >/dev/null; done
for i in {1..10}; do curl -s http://localhost:11434/api/generate \
  -d '{"model":"gemma2:27b","prompt":"warm up","stream":false}' >/dev/null; done
for i in {1..10}; do curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:14b","prompt":"warm up","stream":false}' >/dev/null; done