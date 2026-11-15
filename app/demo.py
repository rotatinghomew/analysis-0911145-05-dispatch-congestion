import json
import threading

from app.client import analysis, request
from app.server import make_server


def main():
    with make_server() as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = 'http://127.0.0.1:' + str(server.server_port)
        print(json.dumps(request(base, '/health'), ensure_ascii=False))
        for case in request(base, '/cases')['cases']:
            response = request(base, '/preview', {'case': case})
            print(json.dumps({'case': case, 'preview': response}, ensure_ascii=False))
        for case in request(base, '/study-cases')['cases']:
            response = request(base, '/study-preview', {'case': case})
            print(json.dumps({'study': case, 'preview': response}, ensure_ascii=False))
        for case in request(base, '/study-cases')['cases']:
            result = analysis(base, case)
            summary = {
                'scope_definition': result['scope_definition'],
                'estimable': result['estimable'],
                'constraints': result['constraints'],
                'conclusions': result['conclusions'],
                'sensitivity': result['sensitivity'],
                'validation': result['validation'],
                'diagnostics': result['diagnostics'],
            }
            print(json.dumps({'cohort_effects': case, 'summary': summary}, ensure_ascii=False))
        server.shutdown()
        worker.join(timeout=3)


if __name__ == '__main__':
    main()
