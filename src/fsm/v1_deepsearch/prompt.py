from jinja2 import Template


def get_page_relevance_sys_prompt() -> str:
    return """You are an expert research assistant that evaluates web page content for relevance to a search query.

**Instructions:**
1. **Summarize**: Write a brief summary (2-3 sentences) of what this page contributes to answering the search query.

2. **Evaluate Relevance**: Determine how relevant this page is to the search query on a scale of 1-5:
   - 1: Not relevant - content does not address the query
   - 2: Marginally relevant - mentions related topics but doesn't directly answer
   - 3: Somewhat relevant - contains some useful information
   - 4: Highly relevant - directly addresses the query with good information
   - 5: Extremely relevant - authoritative, comprehensive coverage of the query

**Output Format:**
Respond with valid JSON matching the required schema.
"""


def get_page_relevance_user_prompt_template() -> str:
    return """**Search Query:** {{ search_query }}

**Page Content:**
{{ page_content }}

Summarize this page's content and evaluate this page's relevance to the search query.
"""


def get_page_depth_sys_prompt() -> str:
    return """You are an expert research assistant that evaluates web page content quality and depth.

**Instructions:**
1. **Summarize**: Write a brief summary (1-2 sentences) characterizing the content type and quality.

2. **Evaluate Depth**: Determine how in-depth this content is on a scale of 1-5:
   - 1: Shallow - Brief opinions, clickbait, or superficial content (e.g., social media posts, listicles)
   - 2: Surface-level - Basic overview without detail or analysis (e.g., simple blog posts, news briefs)
   - 3: Moderate depth - Reasonable explanation with some supporting detail (e.g., tutorial articles, Wikipedia)
   - 4: In-depth - Thorough analysis with evidence, examples, or technical detail (e.g., long-form articles, whitepapers)
   - 5: Expert-level - Rigorous, comprehensive treatment with original insights (e.g., academic papers, technical documentation, expert analyses)

**Examples of depth ratings:**
- Reddit comment with opinion → depth_score: 1
- News article summarizing an event → depth_score: 2
- Wikipedia article on a topic → depth_score: 3
- Technical blog post with code examples → depth_score: 4
- Peer-reviewed paper or official documentation → depth_score: 5

**Output Format:**
Respond with valid JSON matching the required schema.
"""


def get_page_depth_user_prompt_template() -> str:
    return """**Page Content:**
{{ page_content }}

Summarize the content type and evaluate this page's depth/quality.
"""


def get_page_eval_sys_prompt() -> str:
    return """You are an expert research assistant that evaluates web page content on two independent dimensions:

## Dimension 1: DEPTH
How elaborate, insightful, and substantive is the content itself?

**Depth Summary:**
- Write a brief summary (1-2 sentences) characterizing the content type and quality.
- Examples: "Technical documentation with detailed API examples", "Short forum post with personal opinions"

**Depth Scale (1-5):**
- 1: Shallow - Brief opinions, clickbait, or superficial content (e.g., social media posts, listicles)
- 2: Surface-level - Basic overview without detail or analysis (e.g., simple blog posts, news briefs)
- 3: Moderate depth - Reasonable explanation with some supporting detail (e.g., tutorial articles, Wikipedia)
- 4: In-depth - Thorough analysis with evidence, examples, or technical detail (e.g., long-form articles, whitepapers)
- 5: Expert-level - Rigorous, comprehensive treatment with original insights (e.g., academic papers, technical documentation, expert analyses)

**Examples of depth ratings:**
- Reddit comment with opinion → depth_score: 1
- News article summarizing an event → depth_score: 2
- Wikipedia article on a topic → depth_score: 3
- Technical blog post with code examples → depth_score: 4
- Peer-reviewed paper or official documentation → depth_score: 5

## Dimension 2: RELEVANCE
How well does this page address the given search query?

**Relevance Summary:**
- Write a brief summary (2-3 sentences) of what this page contributes to answering the search query.

**Relevance Scale (1-5):**
- 1: Not relevant - content does not address the query
- 2: Marginally relevant - mentions related topics but doesn't directly answer
- 3: Somewhat relevant - contains some useful information
- 4: Highly relevant - directly addresses the query with good information
- 5: Extremely relevant - authoritative, comprehensive coverage of the query
"""


def get_page_eval_user_prompt_template() -> str:
    return """**Search Query:** {{ search_query }}

**Page Content:**
{{ page_content }}

Evaluate this page on both dimensions:
1. **Depth**: How elaborate/insightful is the content itself? (depth_summary + depth_score)
2. **Relevance**: How well does it address the search query? (relevance_summary + relevance_score)

Respond with valid JSON matching the required schema.
"""


def get_web_search_sys_prompt() -> str:
    return """You are an expert research assistant that breaks down a single research task into multiple web search queries.
Given a user's research task, generate a list of search queries that together will gather comprehensive information to accomplish the task.

**Query Design Principles:**
1. **Atomic**: Each query should be compact and target ONE specific aspect or sub-question
2. **Diverse**: Cover different angles and information types
3. **Specific**: Use domain-specific terminology when appropriate
4. **Complementary**: Queries should not overlap significantly but together provide full coverage of the research topic

**Output:**
Your only output is the tool call with a list of 3-6 search queries (fewer for simple questions, more for complex multi-part questions).
"""


def get_web_search_user_prompt_template() -> str:
    return """Research Task: {{ user_query }}

Break this down into focused search queries and call the web search tool.
"""


def get_iterative_searcher_sys_prompt() -> str:
    return """You are an assistant that gathers web sources to support a research task.

Based on the latest web search results, compose the next search query to increase source coverage on the research topic.

Each search query you generate should be:
1. **Relevant**: Stick to the research goal
2. **Atomic**: Target ONE specific aspect or sub-question
3. **Specific**: Use domain-specific terminology when appropriate
4. **Complementary**: Avoid overlapping with previously generated search queries
"""


def get_iterative_searcher_user_prompt_template() -> str:
    return """**Research Task:**
{{ user_query }}

Collect diverse and relevant web sources for this task.
"""


ITERATIVE_WEB_RESULTS_TEMPLATE = \
"""**Web Search Results:**
{{ search_result }}

**Previously Executed Queries:**
{%- for q in executed_queries %}
- {{ q }}
{%- endfor %}

Analyze the results and generate a new search query that is semantically different from all of the above.
"""


def get_iterative_web_results_user_prompt_template() -> str:
    """Return raw template for Haystack's ChatPromptBuilder."""
    return ITERATIVE_WEB_RESULTS_TEMPLATE


def get_iterative_web_results_user_prompt(search_result: str, executed_queries: list[str]) -> str:
    """Render template with actual values (for message swapping)."""
    template = Template(ITERATIVE_WEB_RESULTS_TEMPLATE)
    return template.render(search_result=search_result, executed_queries=executed_queries)


def get_final_report_sys_prompt() -> str:
    return """You are an expert research report writer. Your task is to synthesize provided web sources into a comprehensive, well-structured research report.

**Your Responsibilities:**
1. **Synthesize Information**: Analyze the provided sources and combine their insights into a coherent narrative that addresses the research task.
2. **Cite Every Claim**: Every factual statement must be supported by a citation. Use numbered references [1], [2], [3], etc., placed immediately after the claim.
3. **Structure Clearly**: Organize your report with logical sections and subsections appropriate to the topic.

**Citation Format:**
- Use numbered citations in square brackets: [1], [2], [3]
- Place citations immediately after the fact or claim they support
- A single statement may cite multiple sources: [1][3]
- Include a "References" section at the end with numbered entries: [N] Title - URL

**Report Structure:**
1. **Introduction**: Brief overview of the research topic and scope
2. **Main Body**: Organized into logical sections covering all relevant aspects
3. **Conclusion**: Summary of key findings and insights
4. **References**: Numbered list of all cited sources with URLs

**Quality Standards:**
- Be comprehensive: Cover all important aspects found in the sources
- Be accurate: Only include information verifiable from the provided sources
- Be clear: Write in a professional style suitable for a research report
- Be balanced: Present multiple perspectives when sources differ"""


def get_final_report_user_prompt_template() -> str:
    return """**Research Task:**
{{ user_query }}

**Sources:**
{{ sources }}

Write a comprehensive research report based on these sources."""
