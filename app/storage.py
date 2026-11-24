import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


class StorageError(RuntimeError):
    """Raised when the preference store cannot complete an operation."""


@dataclass(frozen=True, slots=True)
class Preference:
    category: str
    value: str


class PreferenceStorage:
    def __init__(self, database_path: Path, *, timeout_seconds: float = 5.0) -> None:
        if str(database_path) == ":memory:":
            raise ValueError("The preference database must be a file on disk")
        self.database_path = database_path
        self.timeout_seconds = timeout_seconds

    def initialize(self) -> None:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection, connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS preferences (
                        user_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, category)
                    )
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            raise StorageError("Could not initialize preference storage") from exc

    def save_preference(self, user_id: str, category: str, value: str) -> Preference:
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO preferences (user_id, category, value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, category) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CASE
                            WHEN preferences.value <> excluded.value THEN CURRENT_TIMESTAMP
                            ELSE preferences.updated_at
                        END
                    """,
                    (user_id, category, value),
                )
        except sqlite3.Error as exc:
            raise StorageError("Could not save preference") from exc
        return Preference(category=category, value=value)

    def get_preferences(self, user_id: str) -> list[Preference]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT category, value
                    FROM preferences
                    WHERE user_id = ?
                    ORDER BY category
                    """,
                    (user_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Could not read preferences") from exc
        return [Preference(category=row["category"], value=row["value"]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
