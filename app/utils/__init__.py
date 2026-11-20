"""Utils package exports."""
from app.utils.url import canonicalize_url, is_same_domain
from app.utils.robots import RobotsChecker
from app.utils.browser import get_browser_pool, close_browser_pool

__all__ = [
    "canonicalize_url",
    "is_same_domain",
    "RobotsChecker",
    "get_browser_pool",
    "close_browser_pool",
]