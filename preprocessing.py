import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lead
from pyspark.sql.types import IntegerType
from pyspark.sql import Window
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.functions import round as Fround


def import_data_into_dataframe(file_location, file_type,
                               infer_schema="false",
                               first_row_is_header="false",
                               delimiter=","):
    """Imports the specified file into a spark dataframe

    Args:
        file_location (str): Path to the file location
        file_type (str): File type
        infer_schema (str, optional): Should pyspark infer schema. Defaults to "false".
        first_row_is_header (str, optional): Is the first row a header. Defaults to "false".
        delimiter (str, optional): The delimiter used in the file. Defaults to ",".

    Returns:
        pyspark.sql.dataframe.DataFrame: The pyspark dataframe
    """

    spark = SparkSession.builder.getOrCreate()

    data_df = spark.read.format(file_type) \
        .option("inferSchema", infer_schema) \
        .option("header", first_row_is_header) \
        .option("sep", delimiter) \
        .load(file_location)

    return data_df


def plot_bar_graph(xlist, ylist, title="Bar graph", xaxis_title="", yaxis_title=""):
    """Helper function to plot a plotly bar graph

    Args:
        xlist (Iterable): Values for the xaxis
        ylist (iterable): Values for the yaxis
        title (str, optional): The title for the plot. Defaults to "Bar graph".
        xaxis_title (str, optional): The title for the xaxis. Defaults to "".
        yaxis_title (str, optional): The title for the yaxis. Defaults to "".
    """

    fig = go.Figure([go.Bar(x=xlist, y=ylist, text=ylist, textposition='outside')])
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5,
                      opacity=0.6)
    fig.update_layout(title_text=title, title_font_size=30, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                      yaxis_showgrid=False, yaxis_visible=False, xaxis_tickmode='linear', xaxis_tick0=0, xaxis_dtick=1)
    fig.show()


def count_null_values_for_each_column(spark_df):
    """Creates a dictionary of the number of nulls in each column

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe for which the nulls need to be counted

    Returns:
        dict: A dictionary with column name as key and null count as value
    """

    null_counts = {}
    for col_ in spark_df.columns:
        null_counts[col_] = spark_df.filter(f"{col_} is null").count()

    return null_counts


def count_empty_strings_for_each_string_column(spark_df):
    """Creates a dictionary of counts of empty strings in columns of type string

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe for which empty strings need to be counted

    Returns:
        dict: A dictionary with column name as key and empty string counts as value
    """

    empty_string_values = dict()
    for string_column in get_columns_of_type(spark_df, 'string'):
        empty_string_values[string_column] = spark_df.where(f"{string_column} is not null").where(
            pyspark_func_to_trim_strings()(spark_df[string_column]) == '').count()

    return empty_string_values


def count_column_types(spark_df):
    """Returns a pandas dataframe containing the datatype and the number of columns of that datatype

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe for which the types are to be counted

    Returns:
        pandas.DataFrame: A pandas dataframe
    """

    return pd.DataFrame(spark_df.dtypes).groupby(1, as_index=False)[0].agg(
        {'count': 'count', 'names': lambda x: " | ".join(set(x))}).rename(columns={1: "type"})


def get_columns_of_type(spark_df, type_name):
    """Returns a list of columns of the specified datatype

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe
        type_name (str): The datatype

    Returns:
        list: A list containing the names of columns
    """
    return list(map(lambda x: x[0], filter(lambda x: x[1] == type_name, spark_df.dtypes)))


def pyspark_func_to_trim_strings():
    return udf(lambda x: x.strip())


def clean_dataset(spark_df):
    """Removes unnecessary rows from a pyspark dataframe

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe

    Returns:
        pyspark.sql.dataframe.DataFrame: The cleaned spark dataframe
    """

    spark_df_new = spark_df.where(pyspark_func_to_trim_strings()(spark_df['userId']) != '')

    return spark_df_new


def get_counts_as_pandas_df(column_to_groupBy, spark_df):
    """Groups by a the given column and returns a count of rows for each value

    Args:
        column_to_groupBy (str): Name of the column to group by
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe

    Returns:
        pandas.DataFrame: A pandas DataFrame with counts of each value in the given column
    """

    return spark_df.filter(f"{column_to_groupBy} is not null").groupBy([column_to_groupBy]).count().orderBy(
        col('count').desc()).toPandas()


def get_summary_of_category_column(column_name, spark_df, top_n=10, print_result=False):
    """Returns a dataframe of most common values in a column and their counts

    Args:
        column_name (str): The name of the column
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe.=
        top_n (int, optional): The top n number of results. Defaults to 10.
        print_result (bool, optional): Should the result be printed. Defaults to False.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the frequencies
    """

    counts = get_counts_as_pandas_df(column_name, spark_df)

    if print_result:
        print(f"There are {counts.shape[0]} unique {column_name}/s and the most frequent are: ")
        print(counts.head(top_n))

    return counts.head(top_n)


def create_summary_plots(dataframes, titles, rows, cols, super_title, height=1300):
    """Create a plot of all summaries

    Args:
        dataframes (list): A list of the dataframes
        titles (list): A list of titles for each dataframe
        rows (int): Number of rows for the subplot
        cols (int): Number of columns for the subplot
        super_title (string): The main title of the plot
        height (int, optional): The Height of the plot. Defaults to 1300.
    """

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    bar_traces = []

    for cnt_df in dataframes:
        bar_traces.append(
            go.Bar(x=cnt_df.iloc[:, 0].apply(lambda x: x if len(x) < 25 else x[:25] + '...'), y=cnt_df.iloc[:, 1],
                   text=cnt_df.iloc[:, 1], textposition='outside'))

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.add_trace(bar_traces.pop(0), row=i, col=j)
            fig.update_yaxes(visible=False, showgrid=False, row=i, col=j)
            fig.update_xaxes(showgrid=False, row=i, col=j)

    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5,
                      opacity=0.6)
    fig.update_layout(title_text=super_title, title_font_size=50, height=height, showlegend=False)

    fig.show()


def get_counts_for_unique_users(column_name, spark_df, print_result=False):
    """Generates a dataframe having counts for a particular column for unique users

    Args:
        column_name (str): Name of the column
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe
        print_result (bool, optional): Flag to print the result. Defaults to False.

    Returns:
        pandas.DataFrame: Resultant DataFrame
    """

    pd_df = spark_df.select("userId", column_name).distinct().groupBy(column_name).count().toPandas()

    if print_result:
        print(pd_df)

    return pd_df


def spark_func_to_mark_user_cancellation_event():
    return udf(lambda x: 1 if x == "Cancellation Confirmation" else 0)


def add_column_to_flag_cancellation_event(spark_df):
    return spark_df.withColumn("user_cancelled", spark_func_to_mark_user_cancellation_event()("page"))


def add_column_to_mark_rows_for_churned_users(spark_df):
    windowval = Window.partitionBy("userId").orderBy(col("ts").desc()).rangeBetween(Window.unboundedPreceding, 0)

    return spark_df.withColumn('churned', Fsum('user_cancelled').over(windowval)).orderBy("userId", "ts")


def mark_users_as_churners(spark_df):
    """Creates a pyspark dataframe with userId and their corresponding churn nature. 1 if churned, 0 otherwise.

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe containing event logs

    Returns:
        pyspark.sql.dataframe.DataFrame: Resultant pyspark dataframe
    """

    return spark_df.select("userId", "churned").distinct()


def split_into_train_test_80_20(spark_df):
    """Splits the data into train and test of 80-20 split

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe to split

    Returns:
        tuple: A tuple of train and test pyspark dataframes
    """
    users_marked = mark_users_as_churners(spark_df)

    train_users = users_marked.sampleBy("churned", fractions={0: 0.8, 1: 0.8}, seed=10)
    test_users = users_marked.subtract(train_users)

    train_data = spark_df.join(train_users, spark_df.userId == train_users.userId, "inner").drop(
        train_users.userId).drop(train_users.churned)
    test_data = spark_df.join(test_users, spark_df.userId == test_users.userId, "inner").drop(test_users.userId).drop(
        test_users.churned)

    return train_data, train_users, test_data, test_users


def filter_for_event_and_group_by_userId_and_sessionId(event_name, spark_df):
    return spark_df.filter(f'page == "{event_name}"').groupBy("userId", "sessionId")


def get_stat_per_session_for_users(event_name, alias_name, spark_df):
    return filter_for_event_and_group_by_userId_and_sessionId(event_name, spark_df).count().groupBy(
        "userId").mean().select("userId", Fround(col("avg(count)"), 2).alias(alias_name))


def compare_churner_nonchurners(stat_df, title, users_marked_df):
    """Creates a plot to compare a stat between churners and nonchurners

    Args:
        stat_df (pyspark.sql.dataframe.DataFrame): A pyspark dataframe of the stat
        title (str): Title of the plot
        users_marked_df (pyspark.sql.dataframe.DataFrame): A pyspark dataframe indicating users as churners or non churners
        :rtype: None
    """

    joined = users_marked_df.alias("A").join(stat_df.alias("B"), col("A.userId") == col("B.userId"), "left").drop(
        col("B.userId")).toPandas()
    joined.churned = joined.churned.apply(lambda x: 'churner' if x == 1 else "non-churner")
    joined.fillna(0, inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Box(x=joined.iloc[:, 1], y=joined.iloc[:, 2], jitter=0.3, pointpos=-1.8, boxpoints='all',
                         marker_color='rgb(7,40,89)', line_color='rgb(7,40,89)'))
    fig.update_layout(title_text=title, title_font_size=15, width=500, height=500)
    fig.show()


def comparison_summary(event, alias_name, spark_df, users_marked_df, plot=False, plot_title=""):
    """[summary]

    Args:
        event (str): The name of the event
        alias_name (str): Alias name for the generated stat
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe of event logs
        users_marked_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe of marked users
        plot (bool, optional): Option to plot the stat. Defaults to False.
        plot_title (str, optional): Title of the plot. Defaults to "".

    Returns:
        pyspark.sql.dataframe.DataFrame: A pyspark dataframe containing the userId and stat
    """

    stat_summary = get_stat_per_session_for_users(event, alias_name, spark_df)

    if plot:
        stat_summary.show(10)
        compare_churner_nonchurners(stat_summary, plot_title, users_marked_df)

    return stat_summary


def event_count_per_user(event_name, alias_name, spark_df):
    return spark_df.filter(f"page == '{event_name}'").groupBy('userId').count().select("userId",
                                                                                       col("count").alias(alias_name))


def comparison_summary_for_user(event_name, alias_name, spark_df, users_marked_df, plot=False, plot_title=""):
    summary_df = event_count_per_user(event_name, alias_name, spark_df)

    if plot:
        summary_df.show(20)
        compare_churner_nonchurners(summary_df, plot_title, users_marked_df)

    return summary_df


def get_avg_number_of_artists_listened_per_session_per_user(spark_df, users_marked_df, plot=False, plot_title=""):
    """Creates a pyspark dataframe with avg number of artists per session for every user

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe of event logs
        users_marked_df (pyspark.sql.dataframe.DataFrame): The spark dataframe of marked users
        plot (bool, optional): To enable plot. Defaults to False.
        plot_title (str, optional): Plot title. Defaults to "".

    Returns:
        pyspark.sql.dataframe.DataFrame: Pyspark dataframe
    """

    avg_num_of_artists_per_session = spark_df.select("userId", "sessionId", "artist").distinct().groupBy('userId',
                                                                                                         'sessionId').count().groupBy(
        "userId").mean("count").select("userId", Fround(col("avg(count)"), 2).alias("avg_num_of_artists_per_session"))

    if plot:
        avg_num_of_artists_per_session.show(10)
        compare_churner_nonchurners(avg_num_of_artists_per_session, plot_title, users_marked_df)

    return avg_num_of_artists_per_session


def get_number_of_times_each_user_changed_levels(spark_df, users_marked_df, plot=False):
    """Creates a dataframe with number of times a user changed level

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The pyspark dataframe of event logs
        users_marked_df (pyspark.sql.dataframe.DataFrame): The spark dataframe of marked users
        plot (bool, optional): To enable plot. Defaults to False.

    Returns:
        pyspark.sql.dataframe.DataFrame: Pyspark dataframe
    """
    
    winfunc = Window.partitionBy("userId").orderBy('ts')

    num_of_times_user_changed_levels = spark_df.withColumn("leader", lead("level").over(winfunc)).select("userId",
                                                                                                         "level",
                                                                                                         "leader").withColumn(
        "same", col("level") != col("leader")).select("userId", col("same").cast(IntegerType())).groupBy("userId").sum(
        "same").select("userId", col("sum(same)").alias("num_times_user_changed_levels"))

    if plot:
        num_of_times_user_changed_levels.show(10)
        compare_churner_nonchurners(num_of_times_user_changed_levels, "number of times user changed levels",
                                    users_marked_df)

    return num_of_times_user_changed_levels


def get_user_gender(spark_df):
    return spark_df.select("userId", "gender").distinct()


def aggregate_features(spark_df, users_marked_df, enable_plot=False):
    """Creates features for the given dataframe of event logs

    Args:
        spark_df (pyspark.sql.dataframe.DataFrame): The spark dataframe of event logs
        users_marked_df (pyspark.sql.dataframe.DataFrame): The spark dataframe of userId marked as churners or nonchurners
        enable_plot (bool, optional): Option to plot stats. Defaults to False.

    Returns:
        pyspark.sql.dataframe.DataFrame: A pyspark dataframe of the feature matrix
    """

    avg_num_of_songs_per_session = comparison_summary("NextSong", "avg_num_of_songs_per_session", spark_df,
                                                      users_marked_df, enable_plot, "songs per session")

    avg_num_of_adverts_per_session = comparison_summary("Roll Advert", "avg_num_of_adverts_per_session", spark_df,
                                                        users_marked_df, enable_plot, "adverts per session")

    avg_num_of_visits_to_the_home_page_per_session = comparison_summary("Home", "avg_num_of_visits_to_home_per_session",
                                                                        spark_df, users_marked_df,
                                                                        enable_plot,
                                                                        "number of visits to the home page")

    avg_num_of_visits_to_the_about_page_per_session = comparison_summary("About",
                                                                         "average_number_of_visits_to_the_about_page_per_session",
                                                                         spark_df, users_marked_df,
                                                                         enable_plot,
                                                                         "number of visits to the About page per session")

    avg_num_of_visits_to_the_help_page_per_session = comparison_summary("Help",
                                                                        "average_number_of_visits_to_the_help_page_per_session",
                                                                        spark_df, users_marked_df,
                                                                        enable_plot,
                                                                        "number of visits to the Help page per session")

    avg_num_of_visits_to_the_settings_page_per_session = comparison_summary("Settings",
                                                                            "avg_num_of_visits_to_the_settings_page_per_session",
                                                                            spark_df, users_marked_df,
                                                                            enable_plot,
                                                                            "number of visits to the Settings page")

    avg_num_of_times_the_settings_changed_per_session = comparison_summary("Save Settings",
                                                                           "avg_num_of_times_settings_changed_per_session",
                                                                           spark_df, users_marked_df,
                                                                           enable_plot,
                                                                           "number of times settings was changed")

    avg_num_of_thumbs_up_per_session = comparison_summary("Thumbs Up", "avg_num_of_thumbs_up_per_session", spark_df,
                                                          users_marked_df, enable_plot,
                                                          "number of thumbs up")

    avg_num_of_thumbs_down_per_session = comparison_summary("Thumbs Down", "avg_num_of_thumbs_down_per_session",
                                                            spark_df, users_marked_df, enable_plot,
                                                            "number of thumbs down")

    avg_num_of_add_to_playlist_per_session = comparison_summary("Add to Playlist",
                                                                "avg_num_of_add_to_playlist_per_session", spark_df,
                                                                users_marked_df, enable_plot,
                                                                "number of add to playlist")

    avg_num_of_addfriends_per_session = comparison_summary("Add Friend", "avg_num_of_addfriends_per_session", spark_df,
                                                           users_marked_df, enable_plot,
                                                           "number of Add Friend")

    avg_number_of_errors_per_session = comparison_summary("Error", "avg_number_of_errors_per_session", spark_df,
                                                          users_marked_df, enable_plot, "number of Errors")

    avg_num_of_visits_to_upgrade_page = comparison_summary("Upgrade", "avg_num_of_visits_to_upgrade_page", spark_df,
                                                           users_marked_df, enable_plot,
                                                           "number of Upgrade")

    avg_number_of_visits_to_downgrade_page = comparison_summary("Downgrade", "avg_number_of_visits_to_downgrade_page",
                                                                spark_df, users_marked_df, enable_plot,
                                                                "number of downgrades")

    number_of_downgrade_submits_per_user = comparison_summary_for_user("Submit Downgrade",
                                                                       "num_of_downgrades_submitted",
                                                                       spark_df,
                                                                       users_marked_df,
                                                                       enable_plot,
                                                                       "number of downgrades submitted")

    number_of_upgrade_submits_per_user = comparison_summary_for_user("Submit Upgrade", "num_of_upgrades_submitted",
                                                                     spark_df,
                                                                     users_marked_df,
                                                                     enable_plot, "number of upgrades submitted")

    avg_num_of_artists_per_session = get_avg_number_of_artists_listened_per_session_per_user(spark_df,
                                                                                             users_marked_df,
                                                                                             plot=enable_plot,
                                                                                             plot_title="Number of artists per session")

    num_of_times_user_changed_levels = get_number_of_times_each_user_changed_levels(spark_df,
                                                                                    users_marked_df,
                                                                                    plot=enable_plot)

    users_gender = get_user_gender(spark_df)

    final = users_marked_df.alias("A").join(avg_num_of_add_to_playlist_per_session.alias("B"),
                                            col("A.userId") == col("B.userId"), "left").drop(col("B.userId"))
    final = final.join(avg_num_of_addfriends_per_session, final.userId == avg_num_of_addfriends_per_session.userId,
                       "left").drop(avg_num_of_addfriends_per_session.userId)
    final = final.join(avg_num_of_adverts_per_session, final.userId == avg_num_of_adverts_per_session.userId,
                       'left').drop(avg_num_of_adverts_per_session.userId)
    final = final.join(avg_num_of_artists_per_session, final.userId == avg_num_of_artists_per_session.userId,
                       'left').drop(avg_num_of_artists_per_session.userId)
    final = final.join(avg_num_of_songs_per_session, final.userId == avg_num_of_songs_per_session.userId, 'left').drop(
        avg_num_of_songs_per_session.userId)
    final = final.join(avg_num_of_thumbs_down_per_session, final.userId == avg_num_of_thumbs_down_per_session.userId,
                       'left').drop(avg_num_of_thumbs_down_per_session.userId)
    final = final.join(avg_num_of_thumbs_up_per_session, final.userId == avg_num_of_thumbs_up_per_session.userId,
                       'left').drop(avg_num_of_thumbs_up_per_session.userId)
    final = final.join(avg_num_of_times_the_settings_changed_per_session,
                       final.userId == avg_num_of_times_the_settings_changed_per_session.userId, 'left').drop(
        avg_num_of_times_the_settings_changed_per_session.userId)
    final = final.join(avg_num_of_visits_to_the_about_page_per_session,
                       final.userId == avg_num_of_visits_to_the_about_page_per_session.userId, 'left').drop(
        avg_num_of_visits_to_the_about_page_per_session.userId)
    final = final.join(avg_num_of_visits_to_the_help_page_per_session,
                       final.userId == avg_num_of_visits_to_the_help_page_per_session.userId, 'left').drop(
        avg_num_of_visits_to_the_help_page_per_session.userId)
    final = final.join(avg_num_of_visits_to_the_home_page_per_session,
                       final.userId == avg_num_of_visits_to_the_home_page_per_session.userId, 'left').drop(
        avg_num_of_visits_to_the_home_page_per_session.userId)
    final = final.join(avg_num_of_visits_to_the_settings_page_per_session,
                       final.userId == avg_num_of_visits_to_the_settings_page_per_session.userId, 'left').drop(
        avg_num_of_visits_to_the_settings_page_per_session.userId)
    final = final.join(avg_num_of_visits_to_upgrade_page, final.userId == avg_num_of_visits_to_upgrade_page.userId,
                       'left').drop(avg_num_of_visits_to_upgrade_page.userId)
    final = final.join(avg_number_of_errors_per_session, final.userId == avg_number_of_errors_per_session.userId,
                       'left').drop(avg_number_of_errors_per_session.userId)
    final = final.join(avg_number_of_visits_to_downgrade_page,
                       final.userId == avg_number_of_visits_to_downgrade_page.userId, 'left').drop(
        avg_number_of_visits_to_downgrade_page.userId)
    final = final.join(num_of_times_user_changed_levels, final.userId == num_of_times_user_changed_levels.userId,
                       'left').drop(num_of_times_user_changed_levels.userId)
    final = final.join(number_of_downgrade_submits_per_user,
                       final.userId == number_of_downgrade_submits_per_user.userId, 'left').drop(
        number_of_downgrade_submits_per_user.userId)
    final = final.join(number_of_upgrade_submits_per_user, final.userId == number_of_upgrade_submits_per_user.userId,
                       'left').drop(number_of_upgrade_submits_per_user.userId)
    final = final.join(users_gender, final.userId == users_gender.userId, "left").drop(users_gender.userId)

    final.fillna(0)

    return final
