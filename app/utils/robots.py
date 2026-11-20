from urllib.parse import urlparse, urljoin
from typing import List, Optional
import requests
from loguru import logger
import time


class RobotsParser:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.parsed = False
        self._rules = {}
    
    def parse(self, content: str):
        self.parsed = True
        self._rules = {}
        current_user_agent = None
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip().lower(), parts[1].strip()
            if key == "user-agent":
                current_user_agent = val
                if current_user_agent not in self._rules:
                    self._rules[current_user_agent] = {"disallow": [], "crawl_delay": None, "sitemaps": []}
            elif key == "disallow":
                if current_user_agent and (current_user_agent == "*" or current_user_agent in self._rules):
                    if current_user_agent in self._rules:
                        self._rules[current_user_agent]["disallow"].append(val)
            elif key == "allow":
                pass  # allow is default
            elif key == "crawl-delay":
                try:
                    if current_user_agent and current_user_agent in self._rules:
                        self._rules[current_user_agent]["crawl_delay"] = float(val)
                except:
                    pass
            elif key == "sitemap":
                if current_user_agent and current_user_agent in self._rules:
                    self._rules[current_user_agent]["sitemaps"].append(val)
    
    def can_fetch(self, user_agent: str, url: str) -> bool:
        if not self.parsed:
            return True
        parsed_url = urlparse(url)
        path = parsed_url.path
        user_rules = self._rules.get(user_agent, self._rules.get("*", {"disallow": []}))
        for rule in user_rules.get("disallow", []):
            if rule.startswith("/") and path.startswith(rule):
                return False
        return True
    
    def crawl_delay(self, user_agent: str = "*") -> Optional[float]:
        return self._rules.get(user_agent, {}).get("crawl_delay")
    
    def disallowed_paths(self, user_agent: str = "*") -> List[str]:
        return self._rules.get(user_agent, {}).get("disallow", [])
    
    def sitemaps(self) -> List[str]:
        return self._rules.get("*", {}).get("sitemaps", [])


class RobotsChecker:
    """Robots.txt checker with caching"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.parser = RobotsParser(base_url)
        self._fetched = False
        self._fetch_time = 0
        self._cache_ttl = 3600  # 1 hour
    
    def fetch_and_parse(self) -> bool:
        """Fetch and parse robots.txt"""
        if self._fetched and (time.time() - self._fetch_time) < self._cache_ttl:
            return True
        
        try:
            robots_url = urljoin(self.base_url, "/robots.txt")
            resp = requests.get(robots_url, timeout=10)
            if resp.status_code == 200:
                self.parser.parse(resp.text)
                self._fetched = True
                self._fetch_time = time.time()
                return True
            return False
        except Exception as e:
            logger.warning(f"robots.txt check failed for {self.base_url}: {e}")
            return False
    
    def can_fetch(self, user_agent: str, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        if not self._fetched:
            self.fetch_and_parse()
        return self.parser.can_fetch(user_agent, url)
    
    def crawl_delay(self, user_agent: str = "*") -> Optional[float]:
        if not self._fetched:
            self.fetch_and_parse()
        return self.parser.crawl_delay(user_agent)
    
    def disallowed_paths(self, user_agent: str = "*") -> List[str]:
        if not self._fetched:
            self.fetch_and_parse()
        return self.parser.disallowed_paths(user_agent)
    
    def sitemaps(self) -> List[str]:
        if not self._fetched:
            self.fetch_and_parse()
        return self.parser.sitemaps()