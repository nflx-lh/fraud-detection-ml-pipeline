from monitoring.monitoring import Monitoring
from monitoring.utils.config import config, ConfigReader
from monitoring.utils.utils import logger


try:
    logger.info(f"Starting data drift")
    monitor = Monitoring(**config.get("monitoring_config"), project_name="demo")
    monitor.run_monitoring()
    logger.info(f"start server using evidently ui --port 8080 --host 0.0.0.0")
except Exception as e:
    logger.exception(f"Unexpected error during data drift")