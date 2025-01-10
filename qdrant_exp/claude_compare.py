from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import time
from typing import List, Dict
import statistics


class QdrantBenchmark:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.vector_size = 768  # Example vector size, adjust as needed

    def generate_test_data(self, num_users: int, vectors_per_user: int) -> Dict:
        """Generate test data for multiple users"""
        test_data = {}
        for user_id in range(num_users):
            vectors = np.random.rand(vectors_per_user, self.vector_size)
            test_data[user_id] = vectors.astype(np.float32)
        return test_data

    def setup_separate_collections(self, test_data: Dict):
        """Create separate collection for each user"""
        for user_id in tqdm(test_data.keys()):
            collection_name = f"user_{user_id}"
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def setup_single_collection(self):
        """Create single collection with user_id filter"""
        self.client.recreate_collection(
            collection_name="single_collection",
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )
        self.client.create_payload_index(
            collection_name="single_collection",
            field_name="user_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )

    def benchmark_separate_collections(self, test_data: Dict) -> Dict:
        """Benchmark operations with separate collections"""
        insert_times = []
        search_times = []

        # Insertion benchmark
        for user_id, vectors in tqdm(test_data.items()):
            collection_name = f"user_{user_id}"
            start_time = time.time()

            points = [
                models.PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={"vector_id": idx}
                )
                for idx, vector in enumerate(vectors)
            ]
            # Batch things
            for i in range(0, len(points), 1000):
                self.client.upsert(
                    collection_name=collection_name,
                    points=points[i:i + 1000]
                )

            insert_times.append(time.time() - start_time)

        # Search benchmark
        for user_id, vectors in tqdm(test_data.items()):
            collection_name = f"user_{user_id}"
            query_vector = np.random.rand(1, self.vector_size)

            start_time = time.time()
            self.client.search(
                collection_name=collection_name,
                query_vector=query_vector[0],
                limit=5
            )
            search_times.append(time.time() - start_time)

        return {
            "insert_times": insert_times,
            "search_times": search_times
        }

    def benchmark_single_collection(self, test_data: Dict) -> Dict:
        """Benchmark operations with single collection and filters"""
        insert_times = []
        search_times = []

        # Insertion benchmark
        start_time = time.time()
        for user_id, vectors in tqdm(test_data.items()):
            points = [
                models.PointStruct(
                    id=idx + (user_id * len(vectors)),
                    vector=vector.tolist(),
                    payload={"user_id": str(user_id), "vector_id": idx}
                )
                for idx, vector in enumerate(vectors)
            ]

            for i in range(0, len(points), 1000):
                self.client.upsert(
                    collection_name="single_collection",
                    points=points[i:i + 1000]
                )
            insert_times.append(time.time() - start_time)
            start_time = time.time()

        # Search benchmark
        for user_id, vectors in tqdm(test_data.items()):
            query_vector = np.random.rand(1, self.vector_size)[0]

            start_time = time.time()
            self.client.search(
                collection_name="single_collection",
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=5
            )
            search_times.append(time.time() - start_time)

        return {
            "insert_times": insert_times,
            "search_times": search_times
        }

    def run_benchmark(self, num_users: int = 10, vectors_per_user: int = 1000):
        """Run complete benchmark comparing both approaches"""
        print(f"Starting benchmark with {num_users} users and {vectors_per_user} vectors per user")

        # Generate test data
        test_data = self.generate_test_data(num_users, vectors_per_user)

        # Benchmark separate collections
        print("\nBenchmarking separate collections approach...")
        self.setup_separate_collections(test_data)
        separate_results = self.benchmark_separate_collections(test_data)

        # Benchmark single collection
        print("\nBenchmarking single collection with filters approach...")
        self.setup_single_collection()
        single_results = self.benchmark_single_collection(test_data)

        # Analysis
        def analyze_times(times: List[float]) -> Dict:
            return {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0
            }

        separate_analysis = {
            "insert": analyze_times(separate_results["insert_times"]),
            "search": analyze_times(separate_results["search_times"])
        }

        single_analysis = {
            "insert": analyze_times(single_results["insert_times"]),
            "search": analyze_times(single_results["search_times"])
        }

        # Print results
        print("\nResults:")
        print("\nSeparate Collections Approach:")
        print(f"Insert - Mean: {separate_analysis['insert']['mean']:.4f}s, "
              f"Median: {separate_analysis['insert']['median']:.4f}s, "
              f"Std Dev: {separate_analysis['insert']['std_dev']:.4f}s")
        print(f"Search - Mean: {separate_analysis['search']['mean']:.4f}s, "
              f"Median: {separate_analysis['search']['median']:.4f}s, "
              f"Std Dev: {separate_analysis['search']['std_dev']:.4f}s")

        print("\nSingle Collection with Filters Approach:")
        print(f"Insert - Mean: {single_analysis['insert']['mean']:.4f}s, "
              f"Median: {single_analysis['insert']['median']:.4f}s, "
              f"Std Dev: {single_analysis['insert']['std_dev']:.4f}s")
        print(f"Search - Mean: {single_analysis['search']['mean']:.4f}s, "
              f"Median: {single_analysis['search']['median']:.4f}s, "
              f"Std Dev: {single_analysis['search']['std_dev']:.4f}s")


if __name__ == "__main__":
    benchmark = QdrantBenchmark()
    benchmark.run_benchmark(num_users=200, vectors_per_user=1000)