"""This is Data Models Module. It defines all data structures that are used in the application.
They are used to standardize the data flow pipeline."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Article:
    """This class shows a raw news article taken from RSS feeds or HTML sources.
    Attributes are:
        - id: Unique identifier of the article.
        - title: Title of the article.
        - content: Content of the article.
        - url: Web link to the news source.
        - source: Name of the news source.
        - published_at: Date and time when the article was published.
        - raw_html: Raw HTML content of the article.
    """
    id: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    raw_html: Optional[str] = None

@dataclass
class ProcessedArticle:
    """It shows a processed article ready for AI, with clean text and no duplicates
    Attributes are:
        - article_id: This is the reference ID which points to the original article.
        - cleaned_content: This is the cleaned content of the article, with no HTML tags or special characters.
        - embedding: This is the vector representation of the cleaned content. It is used for semantic search and AI processing.
        - cluster_id: This is the cluster ID which groups similar articles together. It is used for clustering and topic modeling.
    """
    article_id: str
    cleaned_content: str
    embedding: List[float] = field(default_factory=list)
    cluster_id: Optional[int] = None

@dataclass
class BriefingItem:
    """This class shows a final version AI-generated news summary
    Attributes are:
        - title: AI-generated title.
        - summary: Bullet points of the article.
        - topic: The topic of the article, which is used for categorization and filtering.
        - source_urls: List of original reference links.
        - relevance_score: How well this news matches user interests (0.0 to 1.0).
    """
    title: str
    summary: str
    topic: str
    relevance_score: float
    source_urls: List[str] = field(default_factory=list)

@dataclass
class UserProfile:
    """This class shows a user profile with their interests and preferences.
    Attributes are:
        - user_id: Unique identifier of the user.
        - topics: List of topics the user is interested in.
        - preferred_length: Preferred length of the news summary ('short', 'medium', 'long').
    """
    user_id: str
    topics: List[str] = field(default_factory=list)
    preferred_length: str = 'medium'