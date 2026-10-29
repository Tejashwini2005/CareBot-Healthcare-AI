
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
CONCURRENT_REQUESTS = 50
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "load_test_results.json"

PAYLOAD = {
    "Age": 35,
    "Gender": "Female",
    "Fever": 1,
    "Cough": 1,
    "Fatigue": 1,
    "Breathlessness": 0,
    "Chest_Pain": 0,
    "Nausea": 0,
    "Headache": 0,
    "BMI": 24.5,
    "BP_Systolic": 120,
    "BP_Diastolic": 80,
    "SpO2": 98,
    "Glucose_mg_dL": 95,
    "HbA1c": 5.4,
}


async def send_request(client, semaphore):
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.post("/predict", json=PAYLOAD)
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "success": response.status_code == 200,
                "error": None if response.status_code == 200 else response.text,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status_code": None,
                "latency_ms": latency_ms,
                "success": False,
                "error": str(exc),
            }


async def main():
    limits = httpx.Limits(
        max_connections=CONCURRENT_REQUESTS,
        max_keepalive_connections=CONCURRENT_REQUESTS,
    )
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        limits=limits,
        timeout=timeout,
    ) as client:
        health = await client.get("/health")
        if health.status_code != 200 or not health.json().get("model_loaded"):
            raise RuntimeError(f"API health check failed: {health.text}")

        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        start = time.perf_counter()
        results = await asyncio.gather(
            *[
                send_request(client, semaphore)
                for _ in range(CONCURRENT_REQUESTS)
            ]
        )
        elapsed_seconds = time.perf_counter() - start

    latencies = sorted(result["latency_ms"] for result in results)
    successes = sum(result["success"] for result in results)
    failures = len(results) - successes

    def percentile(values, percentile_value):
        index = max(0, int((percentile_value / 100) * len(values) + 0.999999) - 1)
        return values[min(index, len(values) - 1)]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{BASE_URL}/predict",
        "concurrent_requests": CONCURRENT_REQUESTS,
        "total_requests": len(results),
        "successful_requests": successes,
        "failed_requests": failures,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "throughput_requests_per_second": round(
            len(results) / elapsed_seconds, 2
        ) if elapsed_seconds else 0,
        "average_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
        "max_latency_ms": round(max(latencies), 2),
        "http_status_counts": {
            str(code): sum(r["status_code"] == code for r in results)
            for code in sorted(
                {r["status_code"] for r in results if r["status_code"] is not None}
            )
        },
        "errors": [r["error"] for r in results if r["error"]],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== CareBot Load Test Results ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
