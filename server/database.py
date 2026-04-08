"""
SQL Surgeon — Database Layer.

Creates and manages the SQLite database used for query optimization tasks.
Includes:
  - Schema creation (e-commerce: customers, products, orders, reviews)
  - Realistic data seeding with deterministic randomness
  - Query execution with timing and result capture
"""

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ── Deterministic Data Generation ────────────────────────────────────────────

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "New York", "San Francisco", "London", "Berlin", "Tokyo",
    "Paris", "Sydney", "Toronto", "Singapore", "Dubai",
]

COUNTRIES = {
    "Mumbai": "India", "Delhi": "India", "Bangalore": "India",
    "Chennai": "India", "Hyderabad": "India", "Pune": "India",
    "Kolkata": "India", "Ahmedabad": "India", "Jaipur": "India",
    "Lucknow": "India", "New York": "USA", "San Francisco": "USA",
    "London": "UK", "Berlin": "Germany", "Tokyo": "Japan",
    "Paris": "France", "Sydney": "Australia", "Toronto": "Canada",
    "Singapore": "Singapore", "Dubai": "UAE",
}

CATEGORIES = [
    "Electronics", "Clothing", "Books", "Home & Kitchen",
    "Sports", "Toys", "Beauty", "Automotive", "Garden", "Food",
]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
    "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Ananya", "Diya", "Myra", "Sara", "Aanya",
    "James", "Emma", "Liam", "Olivia", "Noah",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar",
    "Patel", "Reddy", "Mehta", "Joshi", "Nair",
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Wilson", "Moore",
]

PRODUCT_ADJECTIVES = [
    "Premium", "Ultra", "Pro", "Elite", "Classic",
    "Smart", "Eco", "Turbo", "Max", "Lite",
]

PRODUCT_NOUNS = [
    "Laptop", "Headphones", "Camera", "Watch", "Speaker",
    "Tablet", "Phone Case", "Backpack", "Water Bottle", "Charger",
    "T-Shirt", "Sneakers", "Jacket", "Sunglasses", "Hat",
    "Novel", "Cookbook", "Textbook", "Journal", "Planner",
    "Blender", "Lamp", "Cushion", "Mug", "Candle",
]

ORDER_STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL DEFAULT 1,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    review_text TEXT,
    created_at TEXT NOT NULL
);
""".strip()

# Indexes that would make queries fast — we intentionally omit some of them
# so the "slow" queries actually are slow.
ESSENTIAL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_product_id ON orders(product_id);
CREATE INDEX IF NOT EXISTS idx_orders_status_amount ON orders(status, total_amount);
CREATE INDEX IF NOT EXISTS idx_customers_country ON customers(country);
CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city);
CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_customer_id ON reviews(customer_id);
CREATE INDEX IF NOT EXISTS idx_reviews_order_id ON reviews(order_id);
""".strip()


@dataclass
class QueryResult:
    """Result of executing a SQL query."""

    rows: List[Tuple]
    columns: List[str]
    execution_time_ms: float
    row_count: int
    error: Optional[str] = None


class DatabaseManager:
    """
    Manages the in-memory SQLite database for SQL Surgeon.

    Creates a fresh database each time, seeds it with deterministic data,
    and provides query execution with timing.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.conn: Optional[sqlite3.Connection] = None
        self._rng = random.Random(seed)

    def initialize(self) -> None:
        """Create a fresh in-memory database and seed with data."""
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.executescript(SCHEMA_DDL)
        self.conn.executescript(ESSENTIAL_INDEXES)
        self._seed_data()

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def reset(self) -> None:
        """Reset the database to a fresh state."""
        self.close()
        self.initialize()

    def _seed_data(self) -> None:
        """Seed the database with realistic, deterministic data."""
        assert self.conn is not None
        rng = self._rng

        # ── Customers: 20,000 ──
        customers = []
        for i in range(1, 20_001):
            fname = rng.choice(FIRST_NAMES)
            lname = rng.choice(LAST_NAMES)
            city = rng.choice(CITIES)
            country = COUNTRIES[city]
            year = rng.randint(2019, 2024)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            customers.append((
                i, fname, lname,
                f"{fname.lower()}.{lname.lower()}.{i}@email.com",
                city, country,
                f"{year}-{month:02d}-{day:02d}",
            ))
        self.conn.executemany(
            "INSERT INTO customers VALUES (?,?,?,?,?,?,?)", customers
        )

        # ── Products: 1,000 ──
        products = []
        for i in range(1, 1_001):
            adj = rng.choice(PRODUCT_ADJECTIVES)
            noun = rng.choice(PRODUCT_NOUNS)
            cat = rng.choice(CATEGORIES)
            price = round(rng.uniform(5.0, 500.0), 2)
            stock = rng.randint(0, 1000)
            products.append((i, f"{adj} {noun}", cat, price, stock))
        self.conn.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?)", products
        )

        # ── Orders: 200,000 ──
        orders = []
        for i in range(1, 200_001):
            cust_id = rng.randint(1, 20_000)
            prod_id = rng.randint(1, 1_000)
            qty = rng.randint(1, 10)
            total = round(rng.uniform(10.0, 2000.0), 2)
            status = rng.choice(ORDER_STATUSES)
            year = rng.randint(2020, 2024)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            orders.append((
                i, cust_id, prod_id, qty, total, status,
                f"{year}-{month:02d}-{day:02d}",
            ))
        self.conn.executemany(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?)", orders
        )

        # ── Reviews: 100,000 ──
        # Pull valid (order, customer, product) triplets from existing orders
        reviews = []
        for i in range(1, 100_001):
            # Pick a random order from the generated
            order_record = orders[rng.randint(0, len(orders) - 1)]
            order_id = order_record[0]
            cust_id = order_record[1]
            prod_id = order_record[2]
            
            rating = rng.randint(1, 5)
            text = f"Review text for product {prod_id} by customer {cust_id}."
            year = rng.randint(2020, 2024)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            reviews.append((
                i, order_id, cust_id, prod_id, rating, text,
                f"{year}-{month:02d}-{day:02d}",
            ))
        self.conn.executemany(
            "INSERT INTO reviews VALUES (?,?,?,?,?,?,?)", reviews
        )

        self.conn.commit()

    def execute_query(
        self, query: str, timeout_seconds: float = 10.0
    ) -> QueryResult:
        """
        Execute a SQL query and return results with timing.

        Args:
            query: The SQL query to execute.
            timeout_seconds: Maximum execution time before aborting.

        Returns:
            QueryResult with rows, columns, timing, and optional error.
        """
        assert self.conn is not None

        try:
            # Set a progress handler to enforce timeout
            start = time.perf_counter()
            deadline = start + timeout_seconds

            def progress_handler():
                if time.perf_counter() > deadline:
                    return 1  # Abort
                return 0

            self.conn.set_progress_handler(progress_handler, 1000)

            cursor = self.conn.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            elapsed_ms = (time.perf_counter() - start) * 1000

            self.conn.set_progress_handler(None, 0)

            return QueryResult(
                rows=rows,
                columns=columns,
                execution_time_ms=round(elapsed_ms, 3),
                row_count=len(rows),
            )
        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.conn.set_progress_handler(None, 0)
            error_msg = str(e)
            if "interrupted" in error_msg.lower():
                error_msg = f"Query timed out after {timeout_seconds}s"
            return QueryResult(
                rows=[],
                columns=[],
                execution_time_ms=round(elapsed_ms, 3),
                row_count=0,
                error=error_msg,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.conn.set_progress_handler(None, 0)
            return QueryResult(
                rows=[],
                columns=[],
                execution_time_ms=round(elapsed_ms, 3),
                row_count=0,
                error=str(e),
            )

    def get_query_plan(self, query: str) -> str:
        """Get the EXPLAIN QUERY PLAN output for a query."""
        assert self.conn is not None
        try:
            cursor = self.conn.execute(f"EXPLAIN QUERY PLAN {query}")
            lines = []
            for row in cursor.fetchall():
                lines.append(f"  {'--' * row[1]} {row[3]}")
            return "\n".join(lines) if lines else "No plan available"
        except Exception as e:
            return f"Error getting plan: {e}"

    def get_sample_data(self, tables: Optional[List[str]] = None, limit: int = 5) -> str:
        """Get sample rows from each table as JSON."""
        assert self.conn is not None
        if tables is None:
            tables = ["customers", "products", "orders", "reviews"]

        samples: Dict[str, Any] = {}
        for table in tables:
            try:
                cursor = self.conn.execute(f"SELECT * FROM {table} LIMIT {limit}")
                cols = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                samples[table] = [dict(zip(cols, row)) for row in rows]
            except Exception:
                samples[table] = []

        return json.dumps(samples, indent=2, default=str)

    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        assert self.conn is not None
        stats = {}
        for table in ["customers", "products", "orders", "reviews"]:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        return stats

    def get_schema_info(self) -> str:
        """
        Get detailed schema information including tables, columns, and indexes.
        Used by the CHECK_SCHEMA tool for agent verification.
        """
        assert self.conn is not None
        schema_parts = []
        
        # Get all tables
        cursor = self.conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table_name, table_sql in tables:
            if table_name.startswith("sqlite_"):
                continue
            
            schema_parts.append(f"Table: {table_name}")
            schema_parts.append(table_sql)
            
            # Get columns info
            cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
            cols = cursor.fetchall()
            col_details = []
            for col in cols:
                pk = " (PK)" if col[5] else ""
                col_details.append(f"  - {col[1]} {col[2]}{pk}")
            schema_parts.append("\n".join(col_details))
            
            # Get indexes for this table
            cursor = self.conn.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            if indexes:
                schema_parts.append("Indexes:")
                for idx in indexes:
                    idx_name = idx[1]
                    cursor = self.conn.execute(f"PRAGMA index_info({idx_name})")
                    idx_cols = [c[2] for c in cursor.fetchall()]
                    schema_parts.append(f"  - {idx_name} on ({', '.join(idx_cols)})")
            
            schema_parts.append("-" * 40)
            
        return "\n".join(schema_parts)
