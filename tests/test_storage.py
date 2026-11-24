from concurrent.futures import ThreadPoolExecutor

from app.storage import Preference, PreferenceStorage


def make_storage(tmp_path) -> PreferenceStorage:
    storage = PreferenceStorage(tmp_path / "preferences.db")
    storage.initialize()
    return storage


def test_saves_new_preference(tmp_path) -> None:
    storage = make_storage(tmp_path)

    saved = storage.save_preference("user-a", "drink", "чай")

    assert saved == Preference(category="drink", value="чай")
    assert storage.get_preferences("user-a") == [saved]


def test_updates_value_and_repeated_save_does_not_duplicate(tmp_path) -> None:
    storage = make_storage(tmp_path)
    storage.save_preference("user-a", "drink", "чай")

    storage.save_preference("user-a", "drink", "кофе")
    storage.save_preference("user-a", "drink", "кофе")

    assert storage.get_preferences("user-a") == [Preference(category="drink", value="кофе")]
    assert len(storage.get_preferences("user-a")) == 1


def test_users_are_isolated(tmp_path) -> None:
    storage = make_storage(tmp_path)
    storage.save_preference("user-a", "drink", "чай")
    storage.save_preference("user-b", "drink", "вода")

    assert storage.get_preferences("user-a") == [Preference(category="drink", value="чай")]
    assert storage.get_preferences("user-b") == [Preference(category="drink", value="вода")]


def test_data_survives_storage_recreation(tmp_path) -> None:
    database_path = tmp_path / "preferences.db"
    first_storage = PreferenceStorage(database_path)
    first_storage.initialize()
    first_storage.save_preference("user-a", "color", "зелёный")

    second_storage = PreferenceStorage(database_path)
    second_storage.initialize()

    assert second_storage.get_preferences("user-a") == [
        Preference(category="color", value="зелёный")
    ]


def test_parallel_writes_use_separate_connections_and_keep_users_isolated(tmp_path) -> None:
    storage = make_storage(tmp_path)

    def write_preferences(user_id: str, prefix: str) -> None:
        for index in range(20):
            storage.save_preference(user_id, "drink", f"{prefix}-{index}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_preferences, "user-a", "tea"),
            executor.submit(write_preferences, "user-b", "coffee"),
        ]
        for future in futures:
            future.result()

    assert len(storage.get_preferences("user-a")) == 1
    assert len(storage.get_preferences("user-b")) == 1
    assert storage.get_preferences("user-a")[0].value.startswith("tea-")
    assert storage.get_preferences("user-b")[0].value.startswith("coffee-")
