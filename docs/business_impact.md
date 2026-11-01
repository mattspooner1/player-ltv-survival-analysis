# Business Impact: Player Lifetime Value Prediction

An executive summary of findings, recommendations, and estimated ROI for non-technical stakeholders.

---

## Executive Summary

> **Scope note.** The studio, its GBP 2M budget, 500,000 players and 30% early churn are an **illustrative scenario**, not real studio data. The analysis uses a public SaaS subscription dataset (500 accounts) relabelled with gaming terms. The models did not rank churn risk better than chance on the test set (Cox C-index 0.40, GBSA 0.52), so every insight and GBP figure below is a **hypothesis to validate**, not a result.

In the illustrative scenario, a mid-sized mobile gaming studio spends GBP 2M annually acquiring 500,000 players, but 30% churn within 30 days because the User Acquisition team cannot distinguish high-value players from likely churners at the point of acquisition. We built a predictive framework that estimates each player's 180-day value by combining two models: one that predicts **how long** a player will stay, and another that predicts **how much** they will spend. The framework is designed to surface early warning signals of churn, compare acquisition channels by player quality, and support retention targeting. On this dataset the signals it surfaced are weak, so the framework would need to be rerun and validated on real player data before it could guide budget decisions.

---

## Business Challenge

### The Problem

The studio acquires players through five channels (organic search, paid ads, partner promotions, in-game events, and other referrals) at an average cost of GBP 4 per player. Currently, the UA team treats all channels equally in budget allocation. However, not all players are equal: some generate revenue for months while others leave within weeks. Without a way to predict player value early, the team cannot:

- **Adjust acquisition bids** based on expected player quality from each channel
- **Target retention offers** to players most likely to churn (and most worth saving)
- **Evaluate campaign ROI** beyond simple install counts -- knowing cost-per-install is not enough without knowing value-per-install
no
### The Cost of Inaction

If 30% of acquired players churn within 30 days, the studio wastes approximately GBP 600,000 annually on players who never generate meaningful revenue. Even a modest improvement in targeting -- reducing early churn from 30% to 25% -- would save GBP 100,000 in wasted acquisition spend and redirect that budget toward higher-value segments.

---

## Our Approach

We analysed historical data from 500 players across five data sources: player profiles, subscription records, gameplay activity, support interactions, and churn events. From this data, we built 20 behavioural indicators that capture how players engage in their first days and weeks.

Two models work together to produce a single "predicted lifetime value" (pLTV) score for each player:

1. **A retention model** estimates the probability that a player will still be active in 6 months. This model learns from patterns in subscription length, gameplay consistency, and support interactions.

2. **A revenue model** estimates how much a player will spend over 6 months based on their early behaviour patterns -- how quickly they make their first purchase, how many game features they explore, and how their spending changes over time.

The pLTV score is simply: **probability of staying** multiplied by **expected spending**. A player with a 90% chance of staying and GBP 200 expected revenue has a pLTV of GBP 180. A player with a 50% chance and GBP 300 expected revenue has a pLTV of GBP 150 -- despite higher potential spending, their churn risk makes them less valuable on average.

### Performance at a Glance

![KPI summary showing model performance metrics](../outputs/figures/business_kpi_summary.png)

---

## Key Insights

*Each insight below comes from a model that did not generalise on the test set. Read the "What we found" statements as observations in this dataset and the recommended actions as hypotheses to test.*

### Insight 1: Subscription tenure and spending trajectory are the strongest value signals

**What we found:** Players who have maintained their subscription longer and whose spending is increasing over time account for 40% of the revenue model's predictive power. Subscription tenure alone is associated with a 51% reduction in churn risk.

**Why it matters:** These are the players the studio should work hardest to keep. A player whose monthly spending is growing is deepening their engagement -- they are discovering more content, investing in progression, and building habits.

**Recommended action:** Build a "player health score" dashboard that tracks tenure and spending trajectory for all active players. Flag any player whose spending trend turns negative for proactive outreach.

![Feature importance chart showing top predictors of player lifetime value](../outputs/figures/business_feature_importance.png)

**Estimated impact:** Early identification of at-risk high-value players could preserve GBP 50,000 - 150,000 in annual revenue through timely intervention (this assumes a 10-15% retention uplift from targeted campaigns, which is an unsourced assumption and not a result of this analysis).

### Insight 2: Gameplay consistency matters more than total playtime

**What we found:** The variability of a player's daily sessions (measured as the coefficient of variation) is a stronger churn predictor than the total number of sessions. Players with erratic play patterns -- binge sessions followed by days of inactivity -- are 26% more likely to churn than players with consistent daily habits, even if total playtime is similar.

**Why it matters:** This challenges the common assumption that "more play = better retention." A player who logs in for 30 minutes every day is more valuable than one who plays for 5 hours on Saturday and nothing all week. The studio should optimise for habitual engagement, not peak engagement.

**Recommended action:** Redesign daily login rewards and quest systems to incentivise consistent daily engagement rather than long single sessions. Consider push notification timing that reinforces daily habits.

### Insight 3: Churn rates are uniform across subscription tiers

**What we found:** Free-to-Play, Premium, and VIP players all churn at approximately the same rate (~22%). Subscription tier alone does not predict retention.

**Why it matters:** This means tier-specific retention strategies (e.g., "focus all retention efforts on VIP players") would be ineffective. Behavioural signals -- how players actually engage with the game -- matter far more than what tier they have purchased.

**Recommended action:** Base retention targeting on behavioural risk scores rather than subscription tier. A Free-to-Play player with strong engagement signals may be more worth retaining (as a future conversion candidate) than a VIP player showing disengagement.

![Survival curves by subscription tier showing similar retention across tiers](../outputs/figures/business_survival_by_tier.png)

**Caveat:** This finding most likely reflects the synthetic dataset's uniform churn distribution. It says nothing reliable about how real tiers behave, and the behavioural-over-tier recommendation is untested.

### Insight 4: Partner and organic channels show the highest modelled value

**What we found:** Players acquired through partner promotions and organic search generate the highest average 180-day revenue (GBP 1,349 and GBP 1,686 respectively), while event-acquired players generate the lowest (GBP 1,264). Partner-acquired players also have the lowest churn rate (15% vs. 30% for event-acquired).

**Why it matters:** If the studio allocates UA budget equally across channels, it is overinvesting in lower-value channels and underinvesting in the ones that attract longer-lasting players.

**Recommended action (hypothesis):** If the pattern holds on real data, pilot a small shift of UA budget toward organic and partner programmes, with a holdout group to measure incrementality. Do not reallocate on this evidence alone.

![Survival curves by acquisition channel](../outputs/figures/business_survival_by_channel.png)

**Estimated impact:** Reallocating 20% of the GBP 2M annual budget (GBP 400,000) from the lowest-pLTV channel to the highest could generate GBP 100,000 - 300,000 in incremental annual revenue, depending on how much channel-level LTV differences increase with richer production data.

### Insight 5: Feature exploration in the first week signals long-term value

**What we found:** Players who try more of the game's features in their first week have both lower churn risk (13% reduction) and higher predicted revenue (11% of the revenue model's predictive power). Feature diversity is the fourth most important predictor of 180-day revenue.

**Why it matters:** Players who only engage with one game mode or feature are not discovering the full value of the product. Guided onboarding that surfaces additional features could increase both retention and monetisation.

**Recommended action:** Implement a "feature discovery" onboarding flow that introduces 3-5 key game features in the player's first 7 days. Track feature diversity score as a Day-7 health metric for new cohorts. A/B test the onboarding flow against the current experience.

---

## Results and Recommendations

### Model Performance in Plain Language

The retention model does not reliably rank which players are likely to churn: on the held-out test set the best model scored 0.52 where 0.50 is a coin flip, and the Cox model scored 0.40. No calibration check beyond the integrated Brier score was run, so the probability estimates should not be described as well-calibrated. The revenue model (Random Forest) explains 22% of the variation in 180-day spending and beats a cohort-average baseline, but individual predictions are still very uncertain.

The limited signal may come from the small, relabelled dataset (500 players), but that explanation has not been tested.

**The key takeaway:** The pipeline runs end to end and is designed for retraining on new data. The results on this dataset are too weak to support business decisions, and the dollar figures below should be read as scenario arithmetic.

### Business Value Summary

| Opportunity | Mechanism | Estimated Annual Impact | Confidence |
|-------------|-----------|------------------------|------------|
| Retention campaign targeting | Model-selected offers to high-risk, high-value players | GBP 50,000 - 150,000 in preserved revenue | Low (unvalidated assumption; requires A/B test) |
| Channel budget reallocation | Shift 20% from low-pLTV to high-pLTV channels | GBP 100,000 - 300,000 incremental revenue | Low (requires production pLTV data and an incrementality test) |
| Early warning system | Day-7 behavioural scoring for new cohorts | Would shorten time-to-insight if the scores prove predictive | Low (current models do not rank churn better than chance) |
| Reduced wasted spend | Better targeting reduces early churn | GBP 100,000 in saved acquisition cost | Low (depends on targeting precision, currently unproven) |

**Scenario total: GBP 200,000 - 450,000 annually.** This is arithmetic on stated assumptions, not a forecast, and it depends entirely on models that perform much better than these did and on A/B test validation.

![Budget reallocation recommendation by channel](../outputs/figures/business_budget_reallocation.png)

![ROI waterfall for retention campaign](../outputs/figures/business_roi_waterfall.png)

### Specific Next Steps

**Immediate (0-30 days):**

1. Instrument the top 5 behavioural metrics (usage consistency, feature diversity, days to first purchase, session trend, spending trend) in the studio's product analytics platform
2. Begin computing pLTV by acquisition channel on a weekly basis to establish baselines
3. Design the retention offer A/B test: randomly assign high-risk players to treatment (GBP 5 bonus currency) vs. control (no offer)

**Medium-term (30-90 days):**

4. Run the retention A/B test for 8 weeks to measure actual uplift (the 15% assumption must be validated)
5. Pilot a 10% budget reallocation from the lowest-pLTV channel to the highest, with a holdout group to measure incrementality
6. Retrain the pLTV model on production data with game-specific features (progression, social, purchase patterns)

**Long-term (90+ days):**

7. Integrate pLTV scores into the UA bidding system for automated bid adjustments
8. Build a self-service Looker dashboard for the UA team showing pLTV by channel, tier, and cohort
9. Establish quarterly model retraining cadence with automated drift detection

---

## Methodology Note

The analysis uses a statistical technique called survival analysis, which is specifically designed for "time to event" questions -- in this case, "how long until a player churns?" Unlike standard prediction approaches that simply ask "will they churn: yes or no?", survival analysis estimates the probability of a player remaining active at any future point in time. This is the same analytical framework used in medical research (time to recovery), insurance (time to claim), and -- increasingly -- in subscription analytics and player retention.

The approach naturally handles the fact that most players in the dataset have not yet churned. Rather than discarding these players or making assumptions about their future, survival analysis uses all available information: if a player has been active for 200 days and counting, the model knows they have survived at least 200 days, without guessing when they might leave.

For the full technical specification, see [docs/methodology.md](methodology.md).

---

## Limitations and Caveats

Transparency about limitations builds trust in the analysis and sets realistic expectations.

1. **This analysis was built on a synthetic dataset of 500 players.** The dataset is structurally realistic but does not contain genuine gameplay behaviour. Model performance metrics reflect this limitation. How the framework would perform on production data is unknown.

2. **Churn patterns in the dataset are unusually uniform.** All subscription tiers and most acquisition channels show similar ~22% churn rates. Real gaming data may show more differentiation between segments, which could help, but that is untested.

3. **Revenue predictions have meaningful uncertainty at the individual player level.** The model is more reliable for segment averages (e.g., "one channel's players are worth more than another's on average") than for predicting any single player's exact spend.

4. **ROI estimates are projections, not guarantees.** The retention offer simulation assumes a 15% uplift in retention from a GBP 5 offer. Actual results depend on offer design, player segment, timing, and execution quality. A/B testing is required before scaling any intervention.

5. **Correlation is not causation.** The finding that "players with higher feature diversity churn less" does not necessarily mean that forcing players to try more features will reduce churn. The relationship may reflect an underlying trait (curiosity, engagement) rather than a causal mechanism. Experiments are needed to distinguish correlation from actionable causation.

---

## About This Analysis

**Author:** Matt Spooner

This analysis applies survival analysis to a gaming player analytics scenario using a relabelled public dataset. The statistical framework (Cox Proportional Hazards, Kaplan-Meier estimation) answers the fundamental question behind any retention problem: "who will stay, for how long, and how much value will they generate?"

For technical details, see the [Methodology](methodology.md) document. For feature definitions, see the [Data Dictionary](data_dictionary.md).
