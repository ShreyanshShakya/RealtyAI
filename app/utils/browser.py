from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from contextlib import contextmanager
from loguru import logger
from app.config import get_settings
from typing import Optional

settings = get_settings()


class BrowserPool:
    """Manages a single Playwright browser instance for the scraping session"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._started = False
    
    def start(self):
        """Start the browser if not already started"""
        if self._started:
            return
        
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                ]
            )
            self._context = self._browser.new_context(
                user_agent='RealtyAI/1.0 (+https://github.com/yourusername/realty-ai)',
                viewport={'width': 1280, 'height': 720},
                locale='en-US',
            )
            self._started = True
            logger.info('Browser pool started')
        except Exception as e:
            logger.error(f"Failed to start browser pool: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the browser and cleanup resources"""
        if not self._started:
            return
        
        try:
            if self._context:
                self._context.close()
                self._context = None
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        
        try:
            if self._browser:
                self._browser.close()
                self._browser = None
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        
        try:
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
        
        self._started = False
        logger.info('Browser pool stopped')
    
    @contextmanager
    def page(self) -> Page:
        """Get a new page from the browser context"""
        if not self._started:
            self.start()
        
        if not self._context:
            raise RuntimeError("Browser context not available")
        
        page = self._context.new_page()
        try:
            yield page
        finally:
            try:
                page.close()
            except Exception as e:
                logger.debug(f"Error closing page: {e}")
    
    def fetch(self, url: str, wait_for: str = None, timeout: int = 30000) -> Optional[str]:
        """Fetch a URL using Playwright and return HTML content"""
        if not self._started:
            self.start()
        
        with self.page() as page:
            try:
                page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                if wait_for:
                    page.wait_for_selector(wait_for, timeout=10000)
                return page.content()
            except Exception as e:
                logger.warning(f"Playwright fetch failed for {url}: {e}")
                return None


_browser_pool: Optional[BrowserPool] = None


def get_browser_pool(headless: bool = True) -> BrowserPool:
    """Get or create the global browser pool"""
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool(headless=headless)
    return _browser_pool


def close_browser_pool():
    """Close the global browser pool"""
    global _browser_pool
    if _browser_pool:
        _browser_pool.stop()
        _browser_pool = None