# %%

# ---------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Import analytics engine components
# ---------------------------------------------------------

from partner_insights import analytics_graph, df


# ---------------------------------------------------------
# Configure Streamlit page settings
# ---------------------------------------------------------

st.set_page_config(

    page_title="Partner Insights",

    page_icon="📊",

    layout="centered"
)


# ---------------------------------------------------------
# Apply professional UI styling
# ---------------------------------------------------------

st.markdown(

    '''
    <style>

    .main {

        background-color: #F5F7FA;
    }

    .stApp {

        background-color: #F5F7FA;
    }

    h1 {

        color: #1F3A5F;

        font-size: 42px !important;

        font-weight: 700 !important;
    }

    h2, h3 {

        color: #1F3A5F;
    }

    .info-box {

        background-color: white;

        padding: 18px;

        border-radius: 10px;

        border-left: 5px solid #2E6FBB;

        margin-bottom: 20px;

        box-shadow: 0px 1px 4px rgba(0,0,0,0.08);
    }

    </style>
    ''',

    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Display application title
# ---------------------------------------------------------

st.title(

    "Partner Insights"
)


# ---------------------------------------------------------
# Display application description
# ---------------------------------------------------------

st.markdown(

    '''
    Analyze partner outage data using natural language queries.
    Generate charts, summaries, tables, and operational insights instantly.
    '''
)


# ---------------------------------------------------------
# Display usage instructions
# ---------------------------------------------------------

st.markdown(

    '''
    <div class="info-box">

    <b>How to Use</b><br><br>

    • Ask analytical questions in plain English<br>

    • Request charts such as bar, pie, heatmap, or scatter plots<br>

    • Ask for outage summaries, counts, or partner comparisons<br>

    • Example:
      <i>"Heatmap of outage causes by partner"</i>

    </div>
    ''',

    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Query submission form
# ---------------------------------------------------------

with st.form("analytics_form"):


    # ---------------------------------------------------------
    # User query input
    # ---------------------------------------------------------

    user_query = st.text_input(

        label="Enter analytical query",

        placeholder="Example: Planned vs unplanned outage count by partner"
    )


    # ---------------------------------------------------------
    # Submit button
    # ---------------------------------------------------------

    submitted = st.form_submit_button(

        "Generate Insights"
    )


# ---------------------------------------------------------
# Execute LangGraph workflow
# ---------------------------------------------------------

if submitted:


    # ---------------------------------------------------------
    # Validate empty query
    # ---------------------------------------------------------

    if user_query.strip() == "":

        st.warning(

            "Please enter an analytical query."
        )


    # ---------------------------------------------------------
    # Execute workflow
    # ---------------------------------------------------------

    else:


        # Create workflow input state
        initial_state = {

            "user_query": user_query,

            "dataframe": df
        }


        # ---------------------------------------------------------
        # Clear previous matplotlib figures
        # ---------------------------------------------------------

        plt.close("all")


        # ---------------------------------------------------------
        # Execute LangGraph workflow
        # ---------------------------------------------------------

        with st.spinner(

            "Generating insights..."
        ):

            final_state = analytics_graph.invoke(

                initial_state
            )


        # ---------------------------------------------------------
        # Handle invalid analytical query
        # ---------------------------------------------------------

        if not final_state.get(

            "is_valid_query",

            True
        ):

            st.warning(

                final_state.get(

                    "validation_message",

                    "Please rephrase your query."
                )
            )


        # ---------------------------------------------------------
        # Process successful validation
        # ---------------------------------------------------------

        else:


            # Extract execution result
            execution_result = final_state.get(

                "execution_result",

                {}
            )


            # Extract execution status
            status = execution_result.get(

                "status"
            )


            # ---------------------------------------------------------
            # Handle execution failure
            # ---------------------------------------------------------

            if status != "success":

                st.warning(

                    execution_result.get(

                        "user_message",

                        "Unable to process the request. Please rephrase your query."
                    )
                )


            # ---------------------------------------------------------
            # Render successful execution output
            # ---------------------------------------------------------

            else:


                # Extract workflow result
                result = execution_result.get(

                    "result"
                )


                # ---------------------------------------------------------
                # Render dataframe output
                # ---------------------------------------------------------

                if isinstance(

                    result,

                    pd.DataFrame
                ):


                    # Reset dataframe index for cleaner rendering
                    display_df = result.reset_index()


                    # Render compact table
                    if len(display_df) <= 20:

                        st.table(

                            display_df
                        )


                    # Render larger dataframe
                    else:

                        st.dataframe(

                            display_df,

                            height=400
                        )


                # ---------------------------------------------------------
                # Render KPI output
                # ---------------------------------------------------------

                elif isinstance(

                    result,

                    (
                        int,
                        float,
                        np.number
                    )
                ):

                    st.metric(

                        label="Result",

                        value=result
                    )


                # ---------------------------------------------------------
                # Render chart output
                # ---------------------------------------------------------

                else:


                    # Get matplotlib figure
                    fig = plt.gcf()


                    # Improve chart spacing
                    fig.tight_layout(

                        pad=1.0
                    )


                    # Render chart
                    st.pyplot(

                        fig,

                        width="stretch"
                    )


                    # Close figure after rendering
                    plt.close(fig)



