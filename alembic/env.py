from logging.config import fileConfig
import os

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

# ✅ Import your SQLAlchemy Base
from app.models import Base  # change to your actual model import
from app.core.config import settings  # contains DATABASE_URL

# Load environment variables
load_dotenv()

# Alembic Config object
config = context.config

# Set up logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject your DATABASE_URL dynamically
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Provide metadata for autogeneration
target_metadata = Base.metadata

# Offline migration
def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

# Online migration
def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

# Run migrations
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
