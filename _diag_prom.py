import requests

queries = {
    'node_cpu': '100-(avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))*100)',
    'node_mem': '(1-(node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes))*100',
    'container_cpu': 'sum(rate(container_cpu_usage_seconds_total[5m]))*100',
    'simple_up': 'up',
    'kube_pods': 'kube_pod_info',
    'process_cpu': 'rate(process_cpu_seconds_total[5m])*100',
    'process_mem': 'process_resident_memory_bytes',
}
base = 'http://localhost:9090/api/v1/query'
for name, q in queries.items():
    r = requests.get(base, params={'query': q}, timeout=5)
    data = r.json()
    result = data.get('data', {}).get('result', [])
    print(f'{name}: {len(result)} results')
    if result:
        print(f'  sample: {result[0].get("value", [None, None])[1]}')
