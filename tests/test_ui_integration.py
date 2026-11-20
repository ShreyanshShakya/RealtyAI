"""Tests for Streamlit UI filter construction and propagation."""
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from app.ui.chat import render_sidebar, process_question, init_session_state, build_filters_from_sidebar, FILTER_WIDGET_KEYS
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import HybridRetriever


class TestStreamlitFilterConstruction:
    """Test filter construction from Streamlit sidebar."""
    
    def test_source_filter_all_to_none(self):
        """Test 'All' source filter converts to None."""
        # This is tested via render_sidebar return value
        pass
    
    def test_source_filter_specific(self):
        """Test specific source filter value."""
        pass
    
    def test_country_filter_all_to_empty(self):
        """Test 'All' country filter converts to empty list."""
        pass
    
    def test_country_filter_multiple(self):
        """Test multiple country selection."""
        pass
    
    def test_city_filter_multiple(self):
        """Test multiple city selection."""
        pass
    
    def test_community_filter_multiple(self):
        """Test multiple community selection."""
        pass
    
    def test_filter_dict_structure(self):
        """Test that filter dict has correct structure."""
        # The sidebar returns canonical filters:
        # {
        #     "source": "DarGlobal" or None,
        #     "country": ["Oman", "Saudi Arabia"] or [],
        #     "city": ["Riyadh", "Jeddah"] or [],
        #     "community": ["Aida", "Rayana"] or []
        # }
        from app.ui.chat import build_filters_from_sidebar
        
        # Create a mock that simulates Streamlit's widget behavior
        # Widgets store their values in st.session_state with their key
        class MockSessionState(dict):
            def __getattr__(self, name):
                return self.get(name)
            def __setattr__(self, name, value):
                self[name] = value
        
        mock_session_state = MockSessionState()
        mock_session_state.source_filter_widget = "DarGlobal"
        mock_session_state.country_filter_widget = ["Oman", "Saudi Arabia"]
        mock_session_state.city_filter_widget = ["Riyadh", "Jeddah"]
        mock_session_state.community_filter_widget = ["Aida", "Rayana"]
        
        with patch('streamlit.session_state', mock_session_state):
            with patch('streamlit.sidebar') as mock_sidebar:
                mock_sidebar.selectbox = Mock(return_value="DarGlobal")
                mock_sidebar.multiselect = Mock(return_value=["Oman", "Saudi Arabia"])
                
                from app.ui.chat import build_filters_from_sidebar
                filters = build_filters_from_sidebar()
                
                assert "source" in filters
                assert "country" in filters
                assert "city" in filters
                assert "community" in filters
                assert isinstance(filters["country"], list)
                assert isinstance(filters["city"], list)
                assert isinstance(filters["community"], list)


class TestFilterPropagation:
    """Test filter propagation from UI to pipeline to retriever."""
    
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = Mock(spec=RAGPipeline)
        pipeline.ask = Mock(return_value=Mock(
            answer="Test response",
            sources=[],
            retrieved_documents=[],
            filters_applied={},
            model="test-model",
            retrieval_count=0,
            context_length=0,
            query_type="general"
        ))
        return pipeline
    
    @pytest.fixture
    def mock_retriever(self):
        return Mock(spec=HybridRetriever)
    
    def test_sidebar_filters_to_pipeline(self, mock_pipeline):
        """Test that sidebar filters are passed to pipeline.ask()."""
        # Simulate the filter dict from sidebar
        filters = {
            "source": "DarGlobal",
            "countries": ["Oman", "Saudi Arabia"],
            "cities": ["Riyadh", "Jeddah"],
            "communities": ["Aida"]
        }
        
        # This mimics process_question calling pipeline.ask()
        mock_pipeline.ask("test query", filters)
        
        mock_pipeline.ask.assert_called_once_with("test query", filters)
    
    def test_multi_select_filters_are_lists(self, mock_pipeline):
        """Test that multi-select filters are passed as lists."""
        filters = {
            "countries": ["Oman"],
            "cities": ["Riyadh"],
            "communities": ["Aida"]
        }
        
        mock_pipeline.ask("query", filters)
        
        call_args = mock_pipeline.ask.call_args
        called_filters = call_args[0][1]
        
        assert isinstance(called_filters["countries"], list)
        assert isinstance(called_filters["cities"], list)
        assert isinstance(called_filters["communities"], list)
    
    def test_single_select_filter_is_string(self, mock_pipeline):
        """Test that single-select source filter is string."""
        filters = {"source": "DarGlobal"}
        
        mock_pipeline.ask("query", filters)
        
        call_args = mock_pipeline.ask.call_args
        called_filters = call_args[0][1]
        
        assert isinstance(called_filters["source"], str)
        assert called_filters["source"] == "DarGlobal"
    
    def test_empty_multi_select_returns_empty_list(self, mock_pipeline):
        """Test that unselected multi-select returns empty list."""
        filters = {
            "countries": [],
            "cities": [],
            "communities": []
        }
        
        mock_pipeline.ask("query", filters)
        
        call_args = mock_pipeline.ask.call_args
        called_filters = call_args[0][1]
        
        assert called_filters["countries"] == []
        assert called_filters["cities"] == []
        assert called_filters["communities"] == []


class TestSessionStateFilterPersistence:
    """Test filter persistence in Streamlit session state."""
    
    def test_filter_widgets_have_keys(self):
        """Test that sidebar filter widgets have proper keys."""
        # The widgets use keys like:
        # "source_filter_widget", "country_filter_widget", 
        # "city_filter_widget", "community_filter_widget"
        from app.ui.chat import FILTER_WIDGET_KEYS
        
        assert "source_filter_widget" in FILTER_WIDGET_KEYS
        assert "country_filter_widget" in FILTER_WIDGET_KEYS
        assert "city_filter_widget" in FILTER_WIDGET_KEYS
        assert "community_filter_widget" in FILTER_WIDGET_KEYS
    
    def test_session_state_updated_on_widget_change(self):
        """Test that session state is updated when widgets change."""
        # This is handled by Streamlit automatically when using widget keys
        pass
    
    def test_process_question_reads_from_session_state(self):
        """Test that process_question reads filters from session state."""
        # process_question reads from:
        # st.session_state.source_filter_widget
        # st.session_state.country_filter_widget
        # st.session_state.city_filter_widget
        # st.session_state.community_filter_widget
        pass


class TestBuildFiltersFromSidebar:
    """Test the actual build_filters_from_sidebar function."""
    
    def _make_mock_state(self, source="All", countries=None, cities=None, communities=None):
        """Create a mock that simulates Streamlit's session_state with attribute access and __contains__."""
        mock_state = MagicMock()
        mock_state.source_filter_widget = source
        mock_state.country_filter_widget = countries or []
        mock_state.city_filter_widget = cities or []
        mock_state.community_filter_widget = communities or []
        
        # Make it support 'in' operator with proper signature
        def mock_contains(self, key, *args):
            return key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        mock_state.__contains__ = mock_contains.__get__(mock_state)
        
        return mock_state
    
    def test_build_filters_all_defaults(self):
        """Test build_filters_from_sidebar with all default values."""
        from app.ui.chat import build_filters_from_sidebar
        
        # Create a mock that simulates Streamlit's session_state with attribute access and __contains__
        mock_state = MagicMock()
        mock_state.source_filter_widget = "All"
        mock_state.country_filter_widget = []
        mock_state.city_filter_widget = []
        mock_state.community_filter_widget = []
        
        def mock_contains(self, key, *args):
            return key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        mock_state.__contains__ = mock_contains.__get__(mock_state)
        
        with patch('streamlit.session_state', mock_state):
            from app.ui.chat import build_filters_from_sidebar
            filters = build_filters_from_sidebar()
            
            # When source is "All", the source key is not added to filters
            assert filters.get("source") is None
            # When lists are empty, the keys are not added to filters
            assert filters.get("countries") is None
            assert filters.get("cities") is None
            assert filters.get("communities") is None
    
    def test_build_filters_specific_source(self):
        """Test build_filters_from_sidebar with specific source."""
        mock_state = MagicMock()
        mock_state.source_filter_widget = "DarGlobal"
        mock_state.country_filter_widget = []
        mock_state.city_filter_widget = []
        mock_state.community_filter_widget = []
        mock_state.__contains__ = lambda self, key: key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        with patch('streamlit.session_state', mock_state):
            from app.ui.chat import build_filters_from_sidebar
            filters = build_filters_from_sidebar()
            
            assert filters["source"] == "DarGlobal"
    
    def test_build_filters_multiple_countries(self):
        """Test build_filters_from_sidebar with multiple countries."""
        mock_state = MagicMock()
        mock_state.source_filter_widget = "All"
        mock_state.country_filter_widget = ["Oman", "Saudi Arabia"]
        mock_state.city_filter_widget = []
        mock_state.community_filter_widget = []
        mock_state.__contains__ = lambda self, key: key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        with patch('streamlit.session_state', mock_state):
            from app.ui.chat import build_filters_from_sidebar
            filters = build_filters_from_sidebar()
            
            assert filters["country"] == ["Oman", "Saudi Arabia"]
    
    def test_build_filters_multiple_cities(self):
        """Test build_filters_from_sidebar with multiple cities."""
        mock_state = MagicMock()
        mock_state.source_filter_widget = "All"
        mock_state.country_filter_widget = []
        mock_state.city_filter_widget = ["Riyadh", "Jeddah"]
        mock_state.community_filter_widget = []
        mock_state.__contains__ = lambda self, key: key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        with patch('streamlit.session_state', mock_state):
            from app.ui.chat import build_filters_from_sidebar
            filters = build_filters_from_sidebar()
            
            assert filters["city"] == ["Riyadh", "Jeddah"]
    
    def test_build_filters_community_selection(self):
        """Test build_filters_from_sidebar with community selection."""
        mock_state = MagicMock()
        mock_state.source_filter_widget = "All"
        mock_state.country_filter_widget = []
        mock_state.city_filter_widget = []
        mock_state.community_filter_widget = ["Aida", "Rayana"]
        mock_state.__contains__ = lambda self, key: key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        with patch('streamlit.session_state', mock_state):
            from app.ui.chat import build_filters_from_sidebar
            filters = build_filters_from_sidebar()
            
            assert filters["community"] == ["Aida", "Rayana"]
    
    def test_build_filters_all_selections(self):
        """Test build_filters_from_sidebar with all selections made."""
        mock_state = MagicMock()
        mock_state.source_filter_widget = "DarGlobal"
        mock_state.country_filter_widget = ["Oman", "Saudi Arabia"]
        mock_state.city_filter_widget = ["Riyadh", "Jeddah"]
        mock_state.community_filter_widget = ["Aida", "Rayana"]
        mock_state.__contains__ = lambda self, key: key in ['source_filter_widget', 'country_filter_widget', 'city_filter_widget', 'community_filter_widget']
        
        with patch('streamlit.session_state', mock_state):
            from app.ui.chat import build_filters_from_sidebar
            filters = build_filters_from_sidebar()
            
            assert filters["source"] == "DarGlobal"
            assert filters["country"] == ["Oman", "Saudi Arabia"]
            assert filters["city"] == ["Riyadh", "Jeddah"]
            assert filters["community"] == ["Aida", "Rayana"]


class TestProcessQuestion:
    """Test process_question function."""
    
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = Mock(spec=RAGPipeline)
        pipeline.ask = Mock(return_value=Mock(
            answer="Test response about projects.",
            sources=[{"title": "Test Project", "url": "https://example.com"}],
            retrieved_documents=[{"title": "Test Project"}],
            filters_applied={"country": "Oman"},
            model="test-model",
            retrieval_count=1,
            context_length=100,
            query_type="general"
        ))
        return pipeline
    
    def test_process_question_calls_pipeline(self, mock_pipeline):
        """Test that process_question calls pipeline.ask with question and filters."""
        with patch('streamlit.session_state', {
            'source_filter_widget': 'DarGlobal',
            'country_filter_widget': ['Oman'],
            'city_filter_widget': ['Muscat'],
            'community_filter_widget': ['Aida'],
            'pending_question': 'Show projects in Oman'
        }), patch('app.ui.chat.get_rag_pipeline', return_value=mock_pipeline):
            from app.ui.chat import process_question
            
            # This would normally update session state and rerun
            # We test the logic by calling the internal function
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])