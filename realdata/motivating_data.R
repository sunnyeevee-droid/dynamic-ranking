# Figure 1 and Table 1 (Section 2): descriptive features of the Premier League data,
# 1995/96 to 2019/20.
#
#   Rscript motivating_data.R
#
# Requires R >= 4.1 with ggplot2, patchwork and ragg. Writes
# results/realdata/figures/motivating_data.{pdf,png} and prints the entries of Table 1.

options(device = function(...) ragg::agg_capture(
  width = 7.1, height = 6.1, units = "in", res = 96))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here     <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
repo     <- dirname(here)
out_dir  <- file.path(repo, "results", "realdata", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_stem <- file.path(out_dir, "motivating_data")

first_season <- 1995                 # study period starts in 1995/96
last_season  <- 2019                 # and ends in 2019/20

# Three-letter club codes used in the figure.
club_code <- c(
  "Arsenal" = "ARS", "Aston Villa" = "AVL", "Barnsley" = "BAR",
  "Birmingham City" = "BIR", "Blackburn Rovers" = "BLA", "Blackpool" = "BLP",
  "Bolton Wanderers" = "BOL", "Bournemouth" = "BOU", "Bradford City" = "BRA",
  "Brighton & Hove Albion" = "BHA", "Burnley" = "BUR", "Cardiff City" = "CAR",
  "Charlton Athletic" = "CHA", "Chelsea" = "CHE", "Coventry City" = "COV",
  "Crystal Palace" = "CRY", "Derby County" = "DER", "Everton" = "EVE",
  "Fulham" = "FUL", "Huddersfield Town" = "HUD", "Hull City" = "HUL",
  "Ipswich Town" = "IPS", "Leeds United" = "LEE", "Leicester City" = "LEI",
  "Liverpool" = "LIV", "Manchester City" = "MCI", "Manchester United" = "MUN",
  "Middlesbrough" = "MID", "Newcastle United" = "NEW", "Norwich City" = "NOR",
  "Nottingham Forest" = "NFO", "Portsmouth" = "POR", "Queens Park Rangers" = "QPR",
  "Reading" = "REA", "Sheffield United" = "SHU", "Sheffield Wednesday" = "SHW",
  "Southampton" = "SOU", "Stoke City" = "STK", "Sunderland" = "SUN",
  "Swansea City" = "SWA", "Tottenham Hotspur" = "TOT", "Watford" = "WAT",
  "West Bromwich Albion" = "WBA", "West Ham United" = "WHU", "Wigan Athletic" = "WIG",
  "Wimbledon" = "WIM", "Wolverhampton Wanderers" = "WOL")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
standings <- function(m) {
  hp <- ifelse(m$hg > m$ag, 3, ifelse(m$hg == m$ag, 1, 0))
  ap <- ifelse(m$ag > m$hg, 3, ifelse(m$hg == m$ag, 1, 0))
  tm <- c(m$home, m$away)
  pts <- tapply(c(hp, ap), tm, sum)
  gd  <- tapply(c(m$hg - m$ag, m$ag - m$hg), tm, sum)
  gf  <- tapply(c(m$hg, m$ag), tm, sum)
  d <- data.frame(team = names(pts), pts = as.numeric(pts),
                  gd = as.numeric(gd[names(pts)]), gf = as.numeric(gf[names(pts)]))
  d <- d[order(-d$pts, -d$gd, -d$gf), ]   # league order
  d$pos <- seq_len(nrow(d))
  d
}

games <- read.csv(file.path(repo, "data", "1978_2020_football.csv"), stringsAsFactors = FALSE)
games[] <- lapply(games, function(x) if (is.character(x)) trimws(x) else x)
year    <- as.integer(substr(games$season, 1, 4))
games   <- games[year >= first_season & year <= last_season, ]
stopifnot(all(c(games$home_team, games$away_team) %in% names(club_code)))
games$season_lab <- sub("-", "/", games$season)
season_lab <- sort(unique(games$season_lab))

matches <- setNames(lapply(season_lab, function(s) {
  g <- games[games$season_lab == s, ]
  data.frame(home = unname(club_code[g$home_team]), away = unname(club_code[g$away_team]),
             hg = g$home_goals, ag = g$away_goals)
}), season_lab)
tables  <- lapply(matches, standings)

pop_sd <- function(x) sqrt(mean((x - mean(x))^2))
sea <- data.frame(
  season = season_lab,
  idx    = seq_along(season_lab),
  sd     = vapply(tables, function(d) pop_sd(d$pts), numeric(1)),
  gap    = vapply(tables, function(d) d$pts[1] - d$pts[2], numeric(1))
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
hl      <- c("1995/96" = "#4A90C8", "2011/12" = "#EB8B4B", "2019/20" = "#45A07A")
out_col <- c("Home win" = "#A6CEE3", "Draw" = "#EBEBEB", "Home loss" = "#F2CC65")
ink <- "#222222"; muted <- "#6b6b6b"; grid_col <- "#e6e6e6"; base_col <- "#a3a3a3"

FONT_FAMILY <- "Arial"
update_geom_defaults("text", list(family = FONT_FAMILY))

theme_fig <- theme_classic(base_size = 8, base_family = FONT_FAMILY) +
  theme(
    plot.title          = element_text(size = 9, colour = ink, hjust = 0,
                                       margin = margin(b = 6)),
    plot.title.position = "plot",
    axis.line           = element_line(linewidth = 0.3, colour = muted),
    axis.ticks          = element_line(linewidth = 0.3, colour = muted),
    axis.ticks.length   = unit(2, "pt"),
    axis.text           = element_text(size = 7, colour = muted),
    axis.title          = element_text(size = 8, colour = ink),
    panel.grid.major.y  = element_line(linewidth = 0.25, colour = grid_col),
    legend.position     = "none",
    plot.margin         = margin(4, 8, 4, 4)
  )

panel_title <- function(letter, text) {
  bquote(bold(.(paste0("(", letter, ")"))) ~~ .(text))
}

lims <- c(floor(min(sea$sd)) - 0.5, ceiling(max(sea$sd)) + 0.5)
sd_breaks <- seq(ceiling(lims[1] / 2) * 2, floor(lims[2] / 2) * 2, by = 2)
sea_hl <- subset(sea, season %in% names(hl))

# ---- A: dispersion over time ------------------------------------------------
brk <- seq(1, nrow(sea), by = 4)
pA <- ggplot(sea, aes(idx, sd)) +
  geom_line(colour = base_col, linewidth = 0.45) +
  geom_point(colour = base_col, size = 1.1) +
  geom_point(data = sea_hl, aes(fill = season), shape = 21, colour = "white",
             size = 2.6, stroke = 0.4) +
  scale_fill_manual(values = hl, name = NULL) +
  scale_x_continuous(breaks = brk, labels = sea$season[brk],
                     expand = expansion(add = 0.6)) +
  scale_y_continuous(limits = lims, breaks = sd_breaks) +
  guides(fill = guide_legend(override.aes = list(size = 2.6))) +
  labs(x = "Season", y = "SD of final points",
       title = panel_title("A", "Dispersion of final points")) +
  theme_fig +
  theme(legend.position      = c(0.01, 0.99),   # ggplot2 >= 3.5 may suggest legend.position.inside
        legend.justification = c(0, 1),
        legend.direction     = "horizontal",
        legend.text          = element_text(size = 7, colour = ink),
        legend.key.size      = unit(8, "pt"),
        legend.background    = element_blank())

# ---- B: title-winning margin versus dispersion ------------------------------
pB <- ggplot(sea, aes(sd, gap)) +
  geom_point(colour = base_col, size = 1.6) +
  geom_point(data = sea_hl, aes(fill = season), shape = 21, colour = "white",
             size = 2.6, stroke = 0.4) +
  scale_fill_manual(values = hl) +
  scale_x_continuous(limits = lims, breaks = sd_breaks) +
  scale_y_continuous(breaks = seq(0, 20, 5), limits = c(-0.8, max(sea$gap) + 1)) +
  labs(x = "SD of final points", y = "Title-winning margin",
       title = panel_title("B", "Title-winning margin versus dispersion")) +
  theme_fig +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

# ---- C: 2006/07 match outcomes + six key matches ----------------------------
m0607 <- matches[["2006/07"]]
ord   <- tables[["2006/07"]]$team
key   <- c("MUN", "LIV", "ARS")
cm <- m0607
cm$home <- factor(cm$home, levels = ord)
cm$away <- factor(cm$away, levels = ord)
cm$out  <- factor(sign(cm$hg - cm$ag), levels = c(1, 0, -1), labels = names(out_col))
lab_col  <- ifelse(ord %in% key, ink, "#a0a0a0")
lab_face <- ifelse(ord %in% key, "bold", "plain")

pairs <- data.frame(home = c("MUN", "LIV", "MUN", "ARS", "ARS", "LIV"),
                    away = c("LIV", "MUN", "ARS", "MUN", "LIV", "ARS"))
sc <- m0607[match(paste(pairs$home, pairs$away), paste(m0607$home, m0607$away)), ]
pairs$hg  <- sc$hg
pairs$ag  <- sc$ag
pairs$col <- out_col[c("Home loss", "Draw", "Home win")[sign(pairs$hg - pairs$ag) + 2]]
pairs$y   <- 20 - c(0, 1.25, 3.1, 4.35, 6.2, 7.45)      # rows in matrix units, grouped by pair
legd <- data.frame(lab = names(out_col), col = unname(out_col),
                   y = c(3.5, 2.25, 1.0))
xs <- 21.1                                               # just right of the matrix

pC <- ggplot(cm, aes(away, home, fill = out)) +
  geom_tile(width = 0.95, height = 0.95) +
  scale_fill_manual(values = out_col, drop = FALSE) +
  scale_x_discrete(limits = ord) +
  scale_y_discrete(limits = rev(ord)) +
  annotate("rect", xmin = xs - 0.4, xmax = xs + 0.4, ymin = pairs$y - 0.4,
           ymax = pairs$y + 0.4, fill = pairs$col) +
  annotate("text", x = xs + 0.9, y = pairs$y, hjust = 0, size = 6.5 / .pt, colour = ink, family = FONT_FAMILY,
           label = sprintf("%s  %d-%d  %s", pairs$home, pairs$hg, pairs$ag, pairs$away)) +
  annotate("rect", xmin = xs - 0.4, xmax = xs + 0.4, ymin = legd$y - 0.4,
           ymax = legd$y + 0.4, fill = legd$col) +
  annotate("text", x = xs + 0.9, y = legd$y, hjust = 0, size = 6.5 / .pt, colour = ink, family = FONT_FAMILY,
           label = legd$lab) +
  coord_fixed(xlim = c(0.5, 20.5), ylim = c(0.5, 20.5), expand = FALSE, clip = "off") +
  labs(x = "Away team", y = "Home team",
       title = panel_title("C", "2006/07 match outcomes")) +
  theme_fig +
  theme(axis.line  = element_blank(), axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
        panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),
        # vectorised label styling works but ggplot2 may print a warning
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 6,
                                   colour = lab_col, face = lab_face),
        axis.text.y = element_text(size = 6, colour = rev(lab_col), face = rev(lab_face)),
        plot.margin = margin(4, 70, 4, 4))   # room for the match list

# ---- D: final points of selected seasons ------------------------------------
dd <- do.call(rbind, lapply(names(hl), function(s) {
  t <- tables[[s]]
  data.frame(season = s, team = t$team, pts = t$pts, pos = t$pos,
             grp = ifelse(t$pts == max(t$pts), s, "other"))
}))
dd$season <- factor(dd$season, levels = names(hl))
dd$key    <- paste(dd$team, dd$season, sep = "___")
dd$key    <- factor(dd$key, levels = rev(dd$key[order(dd$season, dd$pos)]))

pD <- ggplot(dd, aes(pts, key, fill = grp)) +
  geom_col(width = 0.72) +
  facet_wrap(~season, nrow = 1, scales = "free_y") +
  scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
  scale_x_continuous(limits = c(0, 105), breaks = c(0, 50, 100), expand = c(0, 0)) +
  scale_fill_manual(values = c(hl, other = "#dcdfe3")) +
  labs(x = "Final points", y = NULL,
       title = panel_title("D", "Final points of selected seasons")) +
  theme_fig +
  theme(axis.line.y        = element_blank(),
        axis.ticks.y       = element_blank(),
        axis.text.y        = element_text(size = 5.8),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(linewidth = 0.25, colour = grid_col),
        strip.background   = element_blank(),
        strip.text         = element_text(size = 7.5, face = "bold", colour = ink),
        panel.spacing.x    = unit(8, "pt"))

# ---------------------------------------------------------------------------
# Assemble and save
# ---------------------------------------------------------------------------
fig <- (pA + pB + plot_layout(widths = c(1.6, 1))) /
       (pC + pD + plot_layout(widths = c(1.15, 1))) +
       plot_layout(heights = c(1, 1.9))

pdf_device <- function(filename, width, height, bg = "white", ...) {
  if (capabilities("aqua")) {
    grDevices::quartz(type = "pdf", file = filename, width = width, height = height,
                       family = FONT_FAMILY, bg = bg)
  } else {
    grDevices::cairo_pdf(filename = filename, width = width, height = height,
                         family = FONT_FAMILY, bg = bg)
  }
}
ggsave(paste0(out_stem, ".pdf"), fig, device = pdf_device,
        width = 7.1, height = 6.1, units = "in", bg = "white")
ggsave(paste0(out_stem, ".png"), fig, device = ragg::agg_png,
        width = 7.1, height = 6.1, units = "in", dpi = 300, bg = "white")

stopifnot(length(matches) == 25L,
          all(vapply(matches, nrow, integer(1)) == 380L),
          sum(vapply(matches, nrow, integer(1))) == 9500L,
          nrow(pairs) == 6L, !anyNA(pairs[c("hg", "ag")]),
          sum(dd$season == "1995/96") == 20L,
          sum(dd$season == "2011/12") == 20L,
          sum(dd$season == "2019/20") == 20L)

# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
draw_share <- tapply(games$result == "D", games$season, mean)
share      <- prop.table(table(factor(games$result, levels = c("H", "D", "A"))))
cat(sprintf("Seasons: %d (%s--%s)\n", length(season_lab), season_lab[1], season_lab[length(season_lab)]))
cat(sprintf("Teams per season: %s\n", paste(unique(vapply(tables, nrow, integer(1))), collapse = ", ")))
cat(sprintf("Matches per season: %s\n", paste(unique(vapply(matches, nrow, integer(1))), collapse = ", ")))
cat(sprintf("Total matches: %d\n", nrow(games)))
cat(sprintf("Distinct clubs: %d\n", length(unique(c(games$home_team, games$away_team)))))
cat(sprintf("Home win / draw / home loss: %.1f%% / %.1f%% / %.1f%%\n",
            100 * share[["H"]], 100 * share[["D"]], 100 * share[["A"]]))
cat(sprintf("Seasonal draw proportion: %.1f%%--%.1f%%\n", 100 * min(draw_share), 100 * max(draw_share)))
