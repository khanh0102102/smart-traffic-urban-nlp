from app.core.config import Settings


def test_settings_build_mysql_url_and_escape_credentials():
    settings = Settings(
        database_url=None,
        db_driver="mysql+pymysql",
        db_host="localhost",
        db_port=3307,
        db_name="stnlp_test",
        db_user="test_user",
        db_password="p@ss/word",
    )

    url = settings.sqlalchemy_url

    assert url.drivername == "mysql+pymysql"
    assert url.host == "localhost"
    assert url.port == 3307
    assert url.database == "stnlp_test"
    assert url.username == "test_user"
    assert url.password == "p@ss/word"
    assert url.query["charset"] == "utf8mb4"


def test_database_url_override_takes_precedence():
    settings = Settings(
        database_url="sqlite:///./test.db",
        db_host="mysql-host-that-should-be-ignored",
    )

    assert settings.sqlalchemy_url.drivername == "sqlite"
    assert settings.sqlalchemy_url.database == "./test.db"
