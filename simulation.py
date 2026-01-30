import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Backgammon Dice Economy Simulator", layout="wide")

st.title("🎲 Backgammon Plus: Growth & Economy Simulator")
st.markdown("""
This simulation allows you to test the dice collection mechanics (**Coupon Collector Problem**) 
and the game economy using various parameters.
""")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("⚙️ Simulation Parameters")

# 1. Simulation Settings
st.sidebar.subheader("1. General Settings")
num_simulations = st.sidebar.slider("Number of Simulations (N)", 1000, 50000, 10000, step=1000)

# 2. Chest Mechanics
st.sidebar.subheader("2. Chest Mechanics")
base_pieces = st.sidebar.number_input("Guaranteed Pieces per Chest", value=2, step=1)
ad_watch_prob = st.sidebar.slider("Ad Watch Rate (%)", 0, 100, 5) / 100
ad_bonus_pieces = st.sidebar.number_input("Bonus Pieces from Ads", value=1, step=1)
daily_chests = st.sidebar.slider("Daily Chests Opened (Player Pace)", 1, 10, 2)

# Average Piece Calculation
avg_pieces_per_chest = base_pieces + (ad_bonus_pieces * ad_watch_prob)
st.sidebar.info(f"📊 Avg. Pieces per Chest: **{avg_pieces_per_chest:.2f}**")

# 3. Dice Face Probabilities
st.sidebar.subheader("3. Dice Face Probabilities")
st.sidebar.caption("Assign weights so that the total equals 100.")
c1, c2, c3 = st.sidebar.columns(3)
f1 = c1.number_input("Face 1", value=20)
f2 = c2.number_input("Face 2 (BottleNeck)", value=10)
f3 = c3.number_input("Face 3", value=20)
f4 = c1.number_input("Face 4", value=20)
f5 = c2.number_input("Face 5 (BottleNeck)", value=10)
f6 = c3.number_input("Face 6", value=20)

# Normalize Probabilities
face_weights = np.array([f1, f2, f3, f4, f5, f6])
face_probs = face_weights / face_weights.sum()

if face_weights.sum() != 100:
    st.sidebar.warning(f"Total weight: {face_weights.sum()} (Automatically normalizing...)")

# 4. Rarity Settings
st.sidebar.subheader("4. Rarity Configuration")

# Common
st.sidebar.markdown("**Common**")
prob_common = st.sidebar.slider("Common Category Chance (%)", 0, 100, 75) / 100
count_common = st.sidebar.number_input("Number of Common Skins", value=3, min_value=1)

# Rare
st.sidebar.markdown("**Rare**")
prob_rare = st.sidebar.slider("Rare Category Chance (%)", 0, 100, 15) / 100
count_rare = st.sidebar.number_input("Number of Rare Skins", value=2, min_value=1)

# Epic
st.sidebar.markdown("**Epic**")
prob_epic = st.sidebar.slider("Epic Category Chance (%)", 0, 100, 10) / 100
count_epic = st.sidebar.number_input("Number of Epic Skins", value=1, min_value=1)

# Total Probability Check
total_prob = prob_common + prob_rare + prob_epic
if abs(total_prob - 1.0) > 0.01:
    st.sidebar.error(f"Total category probability: {total_prob * 100:.0f}%! Please adjust to 100%.")


# --- SIMULATION ENGINE ---
def run_simulation(n_sims, probs):
    """
    Coupon Collector Simulation
    Calculates how many pieces are required to collect all 6 faces with given probabilities.
    """
    results = []
    faces = np.arange(6)

    for _ in range(n_sims):
        collected = set()
        count = 0
        while len(collected) < 6:
            count += 1
            face = np.random.choice(faces, p=probs)
            collected.add(face)
        results.append(count)
    return np.array(results)


# Execution Trigger
if st.sidebar.button("🚀 Start Simulation"):
    with st.spinner('Rolling dice, opening chests...'):

        # 1. Run Core Simulation
        sim_results = run_simulation(num_simulations, face_probs)

        # Statistics
        avg_pieces_needed = np.mean(sim_results)
        p75_pieces_needed = np.percentile(sim_results, 75)
        p90_pieces_needed = np.percentile(sim_results, 90)

        # --- RESULTS PANEL ---

        # A. Key Metrics
        st.header("📊 Analysis Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Pieces Required", f"{avg_pieces_needed:.2f}")
        col2.metric("P75 (75% Guaranteed)", f"{p75_pieces_needed:.0f} Pieces")
        col3.metric("P90 (90% Guaranteed)", f"{p90_pieces_needed:.0f} Pieces")

        # B. Skin Economy Table
        st.subheader("🛠 Time-to-Value (Skin Completion Times)")

        data = []
        rarities = [
            ("Common", prob_common, count_common),
            ("Rare", prob_rare, count_rare),
            ("Epic", prob_epic, count_epic)
        ]

        for name, cat_prob, count in rarities:
            if count > 0 and cat_prob > 0:
                # Specific item drop rate
                specific_prob = cat_prob / count

                # Required Chests & Days (Mean)
                chests_needed_avg = avg_pieces_needed / (avg_pieces_per_chest * specific_prob)
                days_needed_avg = chests_needed_avg / daily_chests

                # Required Chests & Days (P75 - Target Audience)
                chests_needed_p75 = p75_pieces_needed / (avg_pieces_per_chest * specific_prob)
                days_needed_p75 = chests_needed_p75 / daily_chests

                data.append({
                    "Rarity": name,
                    "Specific Drop Rate (%)": f"{specific_prob * 100:.2f}%",
                    "Avg Chests": round(chests_needed_avg, 1),
                    "Avg Days": round(days_needed_avg, 1),
                    "P75 Chests (Target)": round(chests_needed_p75, 1),
                    "P75 Days (Target)": round(days_needed_p75, 1)
                })

        df_results = pd.DataFrame(data)
        st.dataframe(df_results, use_container_width=True)

        # C. Goal Check
        st.subheader("🎯 Goal Analysis: Initial 2 Weeks (Common Skin)")

        target_days = 14
        p75_common_days = df_results[df_results["Rarity"] == "Common"]["P75 Days (Target)"].values[0]

        delta = target_days - p75_common_days
        if delta >= 0:
            st.success(
                f"SUCCESS! P75 players complete their first Common skin in {p75_common_days} days. ({delta:.1f} days ahead of target)")
        else:
            st.error(
                f"FAILED! P75 players take {p75_common_days} days to complete. Target missed by {-delta:.1f} days.")
            st.markdown("**Recommendation:** Increase chest content or equalize face probabilities.")

        # D. Visualization
        st.subheader("📈 Piece Requirement Distribution")
        fig = px.histogram(sim_results, nbins=30,
                           title="Distribution of Total Pieces Required to Complete 6 Faces",
                           labels={'value': 'Pieces Required', 'count': 'Number of Players'})

        fig.add_vline(x=avg_pieces_needed, line_dash="dash", line_color="green", annotation_text="Mean")
        fig.add_vline(x=p75_pieces_needed, line_dash="dash", line_color="red", annotation_text="P75")

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click the button in the sidebar to start the simulation.")
