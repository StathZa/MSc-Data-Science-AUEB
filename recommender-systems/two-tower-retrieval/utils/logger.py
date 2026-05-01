from utils.libs import *

class Logger:
  """Custom Logger class to create an interpretable logger prototype"""
  def __init__(self, *args, **kwargs) -> None:
    self.use_logs: bool = False
    self.loglevel: Union[str, int] = logging.DEBUG if self.use_logs else logging.INFO
    self.log_filename: str = "recommender_system.log"
    self.base_dir: PosixPath = Path(os.getcwd()+"/logs")
    self.logpath: PosixPath = self.base_dir / self.log_filename
    self.log_dir: PosixPath = self.base_dir.mkdir() if not self.base_dir.exists() else self.base_dir
    self.logger = self._build_logger()

  def _build_logger(self):
    """Main function"""
    logger = logging.getLogger("recommender_system_logger")
    logger.setLevel(self.loglevel)

    def handle_exception(exc_type, exc_value, exc_traceback):
      logger.critical(
          "Uncaught exception",
          exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    # Prevent accumulation of concurrent loggers
    if not logger.handlers:
      # Define global formatter
      formatter = logging.Formatter(
          fmt="%(asctime)s %(levelname)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s",
          datefmt="%d-%m-%Y %H:%M:%S"
      )

      file_handler = logging.FileHandler(filename=self.logpath, 
                                         mode="w", encoding="utf-8")
      file_handler.setFormatter(formatter)

      stream_handler = logging.StreamHandler(stream=sys.stderr)
      stream_handler.setFormatter(formatter)
      logger.addHandler(file_handler)
      logger.addHandler(stream_handler)

    return logger