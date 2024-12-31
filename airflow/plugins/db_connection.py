from sqlalchemy import create_engine


class DbConnection:
    def __init__(
        self,
        dbname="postgres",
        user="postgres",
        password="admin",
        host="localhost",
        port=5432
    ):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self._url_object = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        self.engine = create_engine(self._url_object, echo=False)

    def get_storage_url(self) -> str:
        return self._url_object

    @staticmethod
    def airflow_db_connection():
        return DbConnection(
            dbname='msu_bp_restored',
            host='infra-pgsql-1',
            port=5432,
            user='postgres',
            password='admin',
        )

    @staticmethod
    def local_db_connection():
        return DbConnection(
            dbname='msu_bp_restored',
            host='localhost',
            port=5432,
            user='postgres',
            password='admin',
        )

    @staticmethod
    def local_db_optuna_connection():
        return DbConnection(
            dbname='optunadb',
            host='localhost',
            port=5432,
            user='postgres',
            password='admin',
        )
    # postgresql+psycopg2://postgres:admin@localhost:5432/optunadb
