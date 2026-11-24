# ==============================================================================
# STUDENT COST OF LIVING INTELLIGENCE SYSTEM  -  Unit VI: Data Analysis Tool (R)
# Re-implements the core of the Python pipeline in R so every Unit VI topic is covered:
#   reading data & files, vectors/factors, arrays & matrices, lists & data frames,
#   probability distributions, statistical models, manipulating objects, data distribution
# Run from the project folder:  Rscript r/analysis.R
# (Written and checked for syntax, but R was not available in the build sandbox, so run it once locally.)
# ==============================================================================

csv_path <- "data/v1_clean.csv"
dir.create("r/output", showWarnings = FALSE)

# ---- 1. Reading data from files ----------------------------------------------
students <- read.csv(csv_path, stringsAsFactors = FALSE)
str(students)
cat("Rows:", nrow(students), " Columns:", ncol(students), "\n")

# ---- 2. Ordered and unordered factors ----------------------------------------
students$Campus  <- factor(students$Campus)                                   # unordered
students$Housing <- factor(students$Housing, levels = c("Home", "Hostel", "Flat", "PG"))
students$Stress_Level <- factor(cut(students$Stress, breaks = c(0, 3, 6, 10),
                                    labels = c("Low", "Medium", "High")),
                                ordered = TRUE)                                # ordered factor
print(table(students$Housing, students$Stress_Level))
print(levels(students$Stress_Level)); print(students$Stress_Level[1] < "High")

# ---- 3. Vectors, matrices, arrays --------------------------------------------
spend_vec <- students$Rent + students$Food + students$Commute + students$Academics
cost_mat  <- as.matrix(students[, c("Rent", "Food", "Commute", "Academics")])
rownames(cost_mat) <- paste0("S", students$ID)
cat("\nMean vector:\n");     print(colMeans(cost_mat))
cat("\nCovariance matrix:\n"); print(round(cov(cost_mat)))
cat("\nCorrelation matrix:\n"); print(round(cor(cost_mat), 2))
cat("\nMatrix algebra - eigenvalues of covariance (PCA by hand):\n"); print(round(eigen(cov(scale(cost_mat)))$values, 3))
# 3-D array: housing x category x summary statistic
arr <- array(NA, dim = c(4, 4, 2),
             dimnames = list(levels(students$Housing), colnames(cost_mat), c("mean", "sd")))
for (h in levels(students$Housing)) {
  sub <- cost_mat[students$Housing == h, , drop = FALSE]
  arr[h, , "mean"] <- colMeans(sub)
  arr[h, , "sd"]   <- apply(sub, 2, sd)
}
print(round(arr[, , "mean"]))

# ---- 4. Lists and data frames ------------------------------------------------
by_housing <- split(students, students$Housing)                  # list of data frames
summary_list <- lapply(by_housing, function(d) list(
  n = nrow(d), mean_cost = mean(d$Rent + d$Food + d$Commute + d$Academics),
  mean_stress = mean(d$Stress), top_locality = names(sort(table(d$Locality), decreasing = TRUE))[1]))
str(summary_list)
summary_df <- do.call(rbind, lapply(names(summary_list), function(k)
  data.frame(Housing = k, n = summary_list[[k]]$n,
             mean_cost = round(summary_list[[k]]$mean_cost),
             mean_stress = round(summary_list[[k]]$mean_stress, 2))))
print(summary_df)

# ---- 5. Manipulating objects (derived columns, merge, aggregate, apply family) -
students$Computed_Total <- spend_vec
students$Total_Income   <- students$Allowance + students$SideIncome
students$Net_Balance    <- students$Total_Income - students$Computed_Total
students$Deficit        <- as.integer(students$Net_Balance < 0)
agg <- aggregate(cbind(Computed_Total, Stress, Net_Balance) ~ Campus + Housing, data = students, FUN = mean)
print(agg[order(-agg$Computed_Total), ])
tier <- data.frame(Housing = levels(students$Housing), Tier = c("Free", "Low", "High", "High"))
students <- merge(students, tier, by = "Housing")
print(tapply(students$Computed_Total, students$Tier, median))
print(sapply(students[, c("Allowance", "Computed_Total", "Stress")], function(x) c(mean = mean(x), sd = sd(x))))

# ---- 6. Probability distributions in R (d / p / q / r) -----------------------
mu <- mean(students$Computed_Total); s <- sd(students$Computed_Total)
cat("\nP(cost > 25000) under Normal fit:", round(1 - pnorm(25000, mu, s), 3), "\n")
cat("Empirical P(cost > 25000):", mean(students$Computed_Total > 25000), "\n")
cat("90th percentile (qnorm):", round(qnorm(.9, mu, s)), "\n")
p_side <- mean(students$SideIncome > 0)
cat("Binomial P(>=3 of 10 with side income):", round(1 - pbinom(2, 10, p_side), 3), "\n")
set.seed(42)
sim_means <- replicate(5000, mean(sample(students$Computed_Total, 30, replace = TRUE)))
cat("CLT check: SD of sample means =", round(sd(sim_means)), " vs sigma/sqrt(30) =", round(s / sqrt(30)), "\n")

# ---- 7. Data distribution: summaries and plots --------------------------------
print(summary(students[, c("Allowance", "SideIncome", "Computed_Total", "Stress")]))
print(quantile(students$Computed_Total, c(.1, .25, .5, .75, .9)))
png("r/output/distribution.png", width = 1200, height = 500)
par(mfrow = c(1, 3))
hist(students$Computed_Total, breaks = 12, freq = FALSE, col = "grey85", main = "Monthly cost", xlab = "INR")
curve(dnorm(x, mu, s), add = TRUE, lwd = 2, col = "red")
boxplot(Computed_Total ~ Housing, data = students, col = "lightblue", main = "Cost by housing")
qqnorm(students$Computed_Total); qqline(students$Computed_Total, col = "red")
dev.off()

# ---- 8. Statistical models in R -----------------------------------------------
cat("\n== Hypothesis tests ==\n")
print(t.test(Computed_Total ~ Campus, data = students))                       # Welch t
print(var.test(Computed_Total ~ Campus, data = students))                     # F test
print(t.test(Stress ~ I(Rent > 0), data = students, alternative = "less"))    # one-tailed
print(chisq.test(table(students$Housing, students$Stress >= 7)))
print(wilcox.test(Computed_Total ~ Campus, data = students))
cat("\n== ANOVA ==\n")
fit1 <- aov(Computed_Total ~ Housing, data = students); print(summary(fit1)); print(TukeyHSD(fit1))
fit2 <- aov(Computed_Total ~ Campus * Housing, data = students); print(summary(fit2))

cat("\n== Linear regression ==\n")
lm1 <- lm(Computed_Total ~ Allowance + SideIncome + Year + Housing + Campus, data = students)
print(summary(lm1)); print(round(confint(lm1)))
par(mfrow = c(2, 2)); plot(lm1)

cat("\n== Logistic regression: high financial stress ==\n")
students$High_Stress <- as.integer(students$Stress >= 7)
glm1 <- glm(High_Stress ~ Housing + Deficit + log(Total_Income), data = students, family = binomial)
print(summary(glm1)); print(round(exp(cbind(OR = coef(glm1), confint.default(glm1))), 2))

cat("\n== Unsupervised: k-means + PCA ==\n")
X <- scale(students[, c("Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Stress")])
pc <- prcomp(X); print(summary(pc))
set.seed(42); km <- kmeans(X, centers = 4, nstart = 50); print(table(km$cluster))
hc <- hclust(dist(X), method = "ward.D2"); print(table(cutree(hc, 4)))
png("r/output/clusters.png", width = 900, height = 600)
plot(pc$x[, 1:2], col = km$cluster, pch = 19, main = "K-means clusters on PC1/PC2"); dev.off()

write.csv(students, "r/output/students_r_processed.csv", row.names = FALSE)
cat("\nDone. Outputs in r/output/\n")
