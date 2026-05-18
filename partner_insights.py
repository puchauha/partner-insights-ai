# %%
# Import pandas for tabular data processing
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import matplotlib for visualizations
import matplotlib.pyplot as plt

# Import seaborn for advanced charts
import seaborn as sns

# Import typing utilities
from typing import TypedDict, Dict, Any

# Import notebook display helper
from IPython.display import display

# Import LangGraph workflow classes
from langgraph.graph import StateGraph, END

# Import OpenAI chat model
from langchain_openai import ChatOpenAI

#Import Streamlit for interactive UI
import streamlit as st


# %%
# Display all dataframe columns
pd.set_option("display.max_columns", None)

# Configure dataframe width
pd.set_option("display.width", 180)

# Default figure size for charts
DEFAULT_FIGSIZE = (10, 6)

# %%
# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

# %%
# Function to load Excel file
def load_excel_file(file_path: str) -> pd.DataFrame:

    # Try loading file
    try:

        # Read Excel file
        df = pd.read_excel(file_path)

        # Print load confirmation
        print(f"Loaded file successfully: {file_path}")

        # Print row count
        print(f"Row count: {len(df)}")

        # Return dataframe
        return df

    # Handle loading failure
    except Exception as exc:

        # Print failure message
        print("Failed to load file")

        # Raise original exception
        raise exc

# %%
# Function to normalize datetime columns
def normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Create dataframe copy
    normalized_df = df.copy()

    # Iterate through columns
    for column in normalized_df.columns:

        # Convert column name to lowercase
        column_name = column.lower()

        # Detect date or time columns
        if "date" in column_name or "time" in column_name:

            # Attempt datetime conversion
            try:

                # Convert values to datetime
                normalized_df[column] = pd.to_datetime(
                    normalized_df[column],
                    errors="coerce"
                )

            # Ignore conversion failure
            except Exception:
                continue

    # Return normalized dataframe
    return normalized_df

# %%
# Load Excel file
df = load_excel_file("outages.xlsx")

# Normalize datetime columns
df = normalize_datetime_columns(df)

# %%
# Define workflow state
class AnalyticsState(TypedDict, total=False):

    # User query
    user_query: str

    # Working dataframe
    dataframe: pd.DataFrame

    # Validation status
    is_valid_query: bool

    # Validation message
    validation_message: str

    # Detected intent
    intent: str

    # Generated code
    generated_code: str

    # Execution result
    execution_result: Dict[str, Any]

# %%
# Function to validate analytical queries
def build_validation_prompt(
    user_query: str,
    columns: list
) -> str:

    return f"""
You are a strict analytics query validator.

Determine whether the user query is a meaningful
data analytics or visualization request.

Available dataframe columns:
{columns}

VALID query examples:
- outage trend by month
- partner outage count
- average downtime
- heatmap of incidents
- pie chart of outage types
- top 5 partners by duration

INVALID query examples:
- asfafsaf
- hello
- how are you
- tell me a joke
- weather today
- who won IPL
- random text
- open youtube

Important:
If the query is gibberish, random text,
non-analytical or unrelated to dataframe analysis,
return INVALID.

User Query:
{user_query}

Return ONLY:
VALID
or
INVALID
""".strip()

# %%
# Node to validate user query
def validate_query(
    state: AnalyticsState
) -> AnalyticsState:

    # Extract dataframe columns
    dataframe_columns = list(
        state["dataframe"].columns
    )

    # Build validation prompt
    prompt = build_validation_prompt(
        user_query=state["user_query"],
        columns=dataframe_columns
    )

    # Invoke LLM
    response = llm.invoke(prompt)

    # Extract validation response
    validation_result = (
        response.content.strip().upper()
    )

    # Handle invalid query
    if validation_result == "INVALID":

        return {
            **state,
            "is_valid_query": False,
            "validation_message": (
                "Please rephrase your query. "
                "The request does not appear "
                "to be related to the uploaded data."
            )
        }

    # Handle valid query
    return {
        **state,
        "is_valid_query": True
    }

# %%
# Function to build intent classification prompt
def build_intent_prompt(user_query: str) -> str:

    # Return formatted prompt
    return f"""
You are an analytics intent classifier.

Classify the user request into ONLY one category:

- chart
- table
- number

Rules:

- Any graph, chart, visualization, trend,
  heatmap, histogram, scatterplot,
  pie chart, line chart or bar chart
  request is "chart"

- Aggregated tabular summaries are "table"

- Single KPI or metric requests are "number"

User Query:
{user_query}

Return only:
chart
or
table
or
number
""".strip()


# Function to build pandas generation prompt
def build_code_prompt(
    user_query: str,
    columns: list,
    intent: str
) -> str:

    # Return formatted prompt
    return f"""
You are a pandas analytics assistant.

DataFrame name:
df

Available columns:
{columns}

User query:
{user_query}

Intent:
{intent}

Rules:
- Generate Python code only
- Do not generate markdown
- Do not import libraries
- Never call plt.show()
- Store final output in variable named result

Formatting rules:
- For tables, result should contain a DataFrame
- For numbers, result should contain a numeric value

Visualization rules:

- Use seaborn heatmap for heatmaps
- Never use df.plot(kind="heatmap")

- Heatmap workflow:
    1. Create pivot table
    2. Use sns.heatmap()

- Use pandas plot only for:
    line
    bar
    hist
    pie
    scatter

- Use compact professional charts
- Prefer figsize=(6,4)
- Use concise titles
- Avoid oversized labels

Example heatmap code:

heatmap_data = df.pivot_table(
    index='partner_name',
    columns='outage_type',
    values='duration_hours',
    aggfunc='sum',
    fill_value=0
)

plt.figure(figsize=(6,4))

result = sns.heatmap(
    heatmap_data,
    annot=True,
    cmap='Blues'
)

Chart generation strategies:

1. Heatmap
   - Use pivot_table()
   - Use sns.heatmap()

2. Stacked bar chart
   - Use pivot_table()
   - Use plot(kind="bar", stacked=True)

3. Pie chart
   - Use value_counts()

4. Line chart
   - Use grouped aggregations



Scatter plot rules:
- Scatter plots require numeric x and y axes
- Use scatter plots only for numeric relationships
- Never use scatter plot for categorical dimensions
- Do not generate scatter plots using:
    Series.plot.scatter()
- Scatter plots must use:
    DataFrame.plot.scatter()
- If scatter plot is unsuitable for the data,
  generate a bar chart instead
- Rotate labels vertically when categories are dense

Chart readability guidelines:
- Ensure labels are readable
- Rotate labels vertically when categories are dense
- Reduce font size for crowded axes
- Avoid overlapping text
- Dynamically adjust figure size based on category count
- Prefer analytical clarity over default formatting

Visualization readability guidelines:
- If categorical labels are long or crowded:
    - prefer horizontal charts
    - place categories on y-axis instead of x-axis
- Avoid overlapping labels
- Prefer readable layouts over default orientations

Visualization suitability guidelines:
- Avoid scatter plots for long categorical labels
- Prefer bar charts for categorical comparisons
- Prefer horizontal charts when labels are long
- Use scatter plots mainly for numeric relationships

If categorical labels are dense or long:
- prefer horizontal bar charts
- prefer heatmaps
- avoid scatter plots

Result formatting rules:
- For multiple KPI outputs:
    - always return pandas DataFrame
- Do NOT return:
    - tuple
    - list
    - dictionary
- Preferred format:
result = pd.DataFrame
    "Metric": [...],
    "Value": [...]

Filtering and matching rules:
- Use case-insensitive matching for text filters
- Prefer partial string matching instead of exact equality
- For categorical filtering:
    use str.contains()
- Example:
df[
    df["partner_name"]
    .str.contains(
        "oceanic",
        case=False,
        na=False
    )
]

- Avoid strict equality checks for user-provided text filters

Important:
- Never use unstack() unless dataframe has MultiIndex
- Prefer pivot_table() for reshaping

Example stacked bar chart:
pivot_data = df.pivot_table(
    index="partner_name",
    columns="outage_type",
    aggfunc="size",
    fill_value=0
)
plt.figure(figsize=(6,4))
result = pivot_data.plot(
    kind="bar",
    stacked=True
)
""".strip()

# %%
# Define blocked patterns
FORBIDDEN_PATTERNS = [
    "savefig",
    "plt.show",
    "base64",
    "BytesIO"
]


# Function to clean generated code
def clean_generated_code(code: str) -> str:

    # Store cleaned lines
    cleaned_lines = []

    # Iterate through code lines
    for line in code.splitlines():

        # Remove whitespace
        stripped_line = line.strip()

        # Skip import statements
        if stripped_line.startswith("import "):
            continue

        # Skip from import statements
        if stripped_line.startswith("from "):
            continue

        # Check blocked patterns
        blocked = any(
            pattern in stripped_line
            for pattern in FORBIDDEN_PATTERNS
        )

        # Skip blocked content
        if blocked:
            continue

        # Append safe line
        cleaned_lines.append(line)

    # Return cleaned code
    return "\n".join(cleaned_lines)

# %%
# Node to detect analytical intent
def generate_intent(
    state: AnalyticsState
) -> AnalyticsState:
    
    # Skip invalid queries
    if not state.get("is_valid_query", True):
        return state

    # Build intent prompt
    prompt = build_intent_prompt(
        state["user_query"]
    )

    # Invoke LLM
    response = llm.invoke(prompt)

    # Extract response text
    intent = response.content.strip().lower()

    # Validate supported intent
    if intent not in ["chart", "table", "number"]:
        intent = "table"

    # Print intent for debugging
    print(intent)

    # Return updated state
    return {
        **state,
        "intent": intent
    }


# Node to generate pandas code
def generate_pandas_code(
    state: AnalyticsState
) -> AnalyticsState:
    
    # Skip invalid queries
    if not state.get("is_valid_query", True):
        return state

    # Extract dataframe columns
    dataframe_columns = list(
        state["dataframe"].columns
    )

    # Build code generation prompt
    prompt = build_code_prompt(
        user_query=state["user_query"],
        columns=dataframe_columns,
        intent=state["intent"]
    )

    # Invoke LLM
    response = llm.invoke(prompt)

    # Extract generated code
    generated_code = response.content

    # Clean generated code
    generated_code = clean_generated_code(
        generated_code
    )

    # Print generated code
    print(generated_code)

    # Return updated state
    return {
        **state,
        "generated_code": generated_code
    }


# Node to execute generated pandas code
def execute_pandas_code(
    state: AnalyticsState
) -> AnalyticsState:
    
    # Skip invalid queries
    if not state.get("is_valid_query", True):
        return state

    # Close old figures
    plt.close("all")    

    # Create dataframe copy
    df = state["dataframe"].copy()

    # Define execution sandbox
    execution_scope = {

        # Restricted builtins
        "__builtins__": {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round
        },

        # Pandas reference
        "pd": pd,

        # Numpy reference
        "np": np,

        # Matplotlib reference
        "plt": plt,

        # Seaborn reference
        "sns": sns,

        # Working dataframe
        "df": df
    }

    # Execute generated code
    try:

        # Execute python code
        exec(
            state["generated_code"],
            execution_scope
        )

        # Extract result object
        result = execution_scope.get("result")

        # Return successful output
        return {
            **state,
            "execution_result": {
                "status": "success",
                "result": result
            }
        }

    # Handle execution failures
    except Exception as exc:
        #clear partially rendered figures
        plt.close("all")
                
        # Store raw error for debugging 
        raw_error = str(exc)    

        # Default user-safe message
        user_message = (
            "Unable to process the request. "
            "Please rephrase your query."
        )

        # Return failure state
        return {
            **state,
            "execution_result": {
                "status": "failed",
                "error": str(exc),
                "user_message": user_message,
                "debug_error": raw_error
            }
        }

# %%
# Create LangGraph workflow
workflow = StateGraph(
    AnalyticsState
)

# Register validation node
workflow.add_node(
    "validate_query",
    validate_query
)

# Register intent node
workflow.add_node(
    "generate_intent",
    generate_intent
)

# Register code generation node
workflow.add_node(
    "generate_pandas_code",
    generate_pandas_code
)

# Register execution node
workflow.add_node(
    "execute_pandas_code",
    execute_pandas_code
)

# Set entry point
workflow.set_entry_point(
    "validate_query"
)

# Connect validation to intent
workflow.add_edge(
    "validate_query",
    "generate_intent"
)

# Connect intent to code generation
workflow.add_edge(
    "generate_intent",
    "generate_pandas_code"
)

# Connect code generation to execution
workflow.add_edge(
    "generate_pandas_code",
    "execute_pandas_code"
)

# Connect execution to end
workflow.add_edge(
    "execute_pandas_code",
    END
)

analytics_graph = workflow.compile()

# %%
# Function to render execution output
def display_execution_result(
    execution_result: Dict[str, Any]
):
    
    # Extract status
    status = execution_result.get("status")

    # Handle execution failure
    if status != "success":

        # Print failure message
        print("Execution failed")

        # Print detailed error
        print(execution_result.get("error"))    
        print(execution_result.get("user_message"))

        # Stop further processing
        return

    # Extract result
    result = execution_result.get("result")

    # Handle dataframe output
    if isinstance(result, pd.DataFrame):

        # Display dataframe
        display(result)

    # Handle numeric output
    elif isinstance(result, (int, float, np.number)):

        # Print numeric value
        print(f"Result: {result}")

    # Handle visualization output
    else:

        # Attempt chart rendering
        try:

            # Configure chart size
            plt.gcf().set_size_inches(
                DEFAULT_FIGSIZE
            )

            # Render chart
            plt.show()

        # Fallback display
        except Exception:

            # Display raw result
            display(result)

# %%
# Create workflow input
initial_state = {
    "user_query": "partner outage details in a table",
    "dataframe": df
}

# Execute workflow
final_state = analytics_graph.invoke(
    initial_state
)

# Handle invalid analytical queries
print("is_valid_query := ", final_state.get("is_valid_query"))
if not final_state.get(
    "is_valid_query",
    True
):

    print(
        final_state.get(
            "validation_message"
        )
    )

# Handle valid analytical queries
else:

    display_execution_result(
        final_state["execution_result"]
    )


