from utils import *

st.title(
    "Pseudo-continuous Pedotransfer Functions for estimating Soil Water Retention Curve (SWRC)"
)
with st.sidebar:
    st.subheader("Upload your CSV file")
    uploaded_file = st.file_uploader(
        "Make sure columns are named - soil#, clay, silt, sand, BD, and omc",
        type=["csv"],
        help="File should atleast have columns - soil#, clay, silt, and sand",
    )

    st.markdown(
        "Clay [%],  Silt [%], Sand [%], "
        "Bulk Density"
        r" $[cm^3 cm^{-3}]$, "
        "and Organic Matter Content [%]"
    )

st.subheader("Dataset")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("**Glimpse of your dataset**")
    st.write(df)

    if set(["clay", "silt", "sand", "BD", "omc"]).issubset(df.columns):
        st.success("All required columns are present")
        choices = ["model1", "model2", "model3", "model4"]

    elif set(["clay", "silt", "sand", "BD"]).issubset(df.columns):
        st.success("clay, silt, sand, and bulk density(BD) columns are present")
        choices = ["model2", "model3"]

    elif set(["clay", "silt", "sand", "omc"]).issubset(df.columns):
        st.success("clay, silt, sand, and soil organic matter(omc) columns are present")
        choices = ["model2", "model4"]

    elif set(["clay", "silt", "sand"]).issubset(df.columns):
        st.success("clay, silt, and sand columns are present")
        choices = ["model2"]

    else:
        st.error(
            "Please make sure required columns are present and are named correctly"
        )
    st.write(f"Data has {df.shape[0]} rows and {df.shape[1]} columns")

    ## MODEL FITTING
    st.header("Fit the Model")
    model = st.selectbox("Select model", choices)

    # --- Refactored Model Execution ---
    model_configs = {
        "model1": {
            "info": "uses soil texture(SSC), bulk density(BD), and soil organic matter (omc) as inputs",
            "cols": ["soil#", "clay", "silt", "sand", "BD", "omc"],
            "scaler": "ann1_stdscaler.pkl",
            "rosetta_model": None,
        },
        "model2": {
            "info": "uses soil texture(SSC) as input",
            "cols": ["soil#", "clay", "silt", "sand"],
            "scaler": "ann2_stdscaler.pkl",
            "rosetta_model": {"code": 2, "cols": ["sand", "silt", "clay"]},
        },
        "model3": {
            "info": "uses soil texture(SSC), and bulk density(BD) as inputs",
            "cols": ["soil#", "clay", "silt", "sand", "BD"],
            "scaler": "ann3_stdscaler.pkl",
            "rosetta_model": {"code": 3, "cols": ["sand", "silt", "clay", "BD"]},
        },
        "model4": {
            "info": "uses soil texture(SSC), and organic matter content(omc) as inputs",
            "cols": ["soil#", "clay", "silt", "sand", "omc"],
            "scaler": "ann4_stdscaler.pkl",
            "rosetta_model": None,
        },
    }

    config = model_configs[model]
    st.info(f'{model} {config["info"]}')

    models = [f"{model}/ann_{i}.h5" for i in range(100)]
    colList = config["cols"]
    df_group = df[colList].groupby(["soil#"])
    test_df = df_group.apply(create_Xtest).reset_index(drop=True)

    scaler = pickle.load(open(config["scaler"], "rb"))

    estimated_vwc = {}
    ros_dict = {}
    vg_parms = {}

    for soil, soil_test in test_df.groupby("soil#"):
        # PCPTF prediction
        X_test = scaler.transform(soil_test.iloc[:, 1:])
        mean_vwc, std_vwc = bag_predict(models, X_test)
        estimated_vwc[soil] = pd.DataFrame({"mean_vwc": mean_vwc, "std_vwc": std_vwc})

        # Rosetta prediction (if applicable)
        if config["rosetta_model"]:
            rosetta_config = config["rosetta_model"]
            rosetta_input = np.array(soil_test[rosetta_config["cols"]].head(1))
            rosetta_predictor = Rosetta(
                rosetta_version=3, model_code=rosetta_config["code"]
            )
            mean, stdev = rosetta_predictor.predict(rosetta_input)

            vg_parms[soil] = pd.DataFrame(
                mean,
                columns=[
                    "theta_r",
                    "theta_s",
                    "log10(alpha)",
                    "log10(n)",
                    "log10(ksat)",
                ],
            )

            h_cm = 10 ** soil_test["pF"]
            rosetta_vwc = [van_genuchten(h=h, parms=mean[0]) for h in h_cm]
            vwc_std_lower = [van_genuchten(h=h, parms=mean[0] - stdev[0]) for h in h_cm]
            vwc_std_upper = [van_genuchten(h=h, parms=mean[0] + stdev[0]) for h in h_cm]

            ros_dict[soil] = pd.DataFrame(
                {
                    "pF": soil_test["pF"],
                    "h_cm": h_cm,
                    "rosetta": rosetta_vwc,
                    "_std": vwc_std_lower,
                    "std": vwc_std_upper,
                }
            )

    # Consolidate PCPTF results
    vwc_ = (
        pd.concat(estimated_vwc.values(), keys=estimated_vwc.keys())
        .reset_index(level=0)
        .rename({"level_0": "soil#"}, axis=1)
        .reset_index(drop=True)
    )
    results_df = pd.concat([test_df, vwc_.iloc[:, 1:]], axis=1)

    # Consolidate Rosetta results
    if ros_dict:
        rosetta_vwc_df = (
            pd.concat(ros_dict.values(), keys=ros_dict.keys())
            .reset_index(level=0)
            .rename({"level_0": "soil#"}, axis=1)
            .reset_index(drop=True)
        )
        vg_parms_df = (
            pd.concat(vg_parms.values(), keys=vg_parms.keys())
            .reset_index(level=0)
            .rename({"level_0": "soil#"}, axis=1)
            .reset_index(drop=True)
        )
    else:
        rosetta_vwc_df = pd.DataFrame()

    st.markdown("**Results for your data**")
    st.write(results_df)
    if not rosetta_vwc_df.empty:
        st.markdown("**Van Genuchten parameteres for your soils are:**")
        st.write(vg_parms_df)

    st.subheader("Plot Results")
    if st.button(
        "Plot random soil",
        help="SWRC for a random soil from your dataset will be created at each click of this button",
    ):
        fig = plot_results(results_df, rosetta_vwc_df)
        st.pyplot(fig)
        st.markdown(
            "***pF***: the logarithmic transformation of soil tension in ***cm*** of water"
        )

    st.subheader("Download Results")
    csv = results_df.to_csv(index=False)
    st.download_button(
        "Download Results", csv, "PCPTF_results.csv", "text/csv", key="download-csv"
    )

else:
    st.info("Awaiting for CSV file to be uploaded.")
