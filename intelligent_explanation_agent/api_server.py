from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import urlparse

from .agent import ExplainableRecommendationAgent
from .settings import Settings


def create_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    settings = Settings.from_env()
    recommender = settings.make_recommender()
    agent = ExplainableRecommendationAgent(recommender, settings.make_llm(),
                                           settings.recent_limit,
                                           settings.include_user_id_in_api)
    model_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/health":
                    self.send_json(200, {"status": "ok"})
                elif path == "/users":
                    self.send_json(200, {"user_ids": recommender.list_user_ids()})
                elif path.startswith("/recommendations/"):
                    user_id = int(path.rsplit("/", 1)[-1])
                    # PyTorch 推理及首次惰性加载串行化，HTTP/API 解释仍在请求生命周期内完成。
                    with model_lock:
                        result = agent.run(user_id)
                    self.send_json(200, result)
                else:
                    self.send_json(404, {"error": "not_found"})
            except ValueError as exc:
                self.send_json(404, {"error": str(exc)})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    server = create_server()
    print("Explainable recommender listening on http://127.0.0.1:8080")
    server.serve_forever()


if __name__ == "__main__":
    main()
