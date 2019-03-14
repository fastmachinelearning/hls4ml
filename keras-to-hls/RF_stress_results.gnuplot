# Input and output file names
#CSV_FILE = "RF_stress_results_KERAS_3layer.csv"
#PDF_FILE = "RF_stress_results_KERAS_3layer.pdf"
CSV_FILE = "RF_stress_results_2layer_100x100.csv"
PDF_FILE = "RF_stress_results_2layer_100x100.pdf"

# Set Gnuplot output on PDF file
set terminal pdfcairo enhanced dashed
set output PDF_FILE

set size 1.0, 1.0
set origin 0.0, 0.0

# CSV file (comma separated values)
set datafile separator ','
set datafile missing '-1'

# X and Y axes
set xtics 10
set logscale y 2
set xtics font ",4"
set ytics font ",4"

set lmargin 6
set rmargin 3

# Line styles
set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

set style line 2 \
    linecolor rgb '#00ad00' \
    linetype 1 linewidth 1 \
    pointtype 7 pointsize 0.2

set style line 3 \
    linecolor rgb '#ad0000' \
    linetype 1 linewidth 1 \
    pointtype 13 pointsize 0.2

set style line 4 \
    linetype 0 linewidth 0.5 \
    linecolor rgb "#808080"

# Legend
set key font ",6"
#set key below

set grid back linestyle 4


# =============================================================================
# =============================================================================
# PAGE 1
# =============================================================================
# =============================================================================

set multiplot layout 2,2

# =============================================================================
# Success
# =============================================================================
set size 0.5,0.5
set origin 0.0,0.5 # Top-Left
#set xlabel "Success" font ",8"
#set ylabel "II" font ",8"
plot CSV_FILE using 17:($18 < 0 ? NaN : 1) title "Success" with linespoints linestyle 3

# =============================================================================
# HLS Time
# =============================================================================
set size 0.5,0.5
set origin 0.5,0.5 # Top-Right
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "HLS Time (s)" offset 5,0 font ",8"
plot CSV_FILE using 17:18 title "HLS Time (s)" with linespoints linestyle 1

# =============================================================================
# (Best and Worst) Latency
# =============================================================================
set size 0.5,0.5
set origin 0.0,0.0 # Bottom-Left
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "Best Latency" font ",8"
plot CSV_FILE using 17:6 title "Best Latency" with linespoints linestyle 2, \
     CSV_FILE using 17:7 title "Worst Latency" with linespoints linestyle 1

# =============================================================================
# (Min and Max) II
# =============================================================================
set size 0.5,0.5
set origin 0.5,0.0 # Bottom-Right
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "II" font ",8"
plot CSV_FILE using 17:8 title "Min II" with linespoints linestyle 2, \
     CSV_FILE using 17:9 title "Max II" with linespoints linestyle 1

unset multiplot

# =============================================================================
# =============================================================================
# PAGE 2
# =============================================================================
# =============================================================================

set multiplot layout 2,2
#title "Performance/Costs vs. Reuse Factor"

# =============================================================================
# DSP
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.0,0.5 # Top-Left
set key right
#set xlabel "Reuse Factor" font ",8"
#set ylabel "DSP" offset 5,0 font ",8"
plot CSV_FILE using 17:11 title "DSP" with linespoints linestyle 1

# =============================================================================
# FF
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.5,0.5 # Top-Right
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "FF" offset 5,0 font ",8"
plot CSV_FILE using 17:12 title "FF" with linespoints linestyle 1

# =============================================================================
# LUT
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.0,0.0 # Bottom-Left
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "LUT" offset 5,0 font ",8"
plot CSV_FILE using 17:13 title "LUT" with linespoints linestyle 1

# =============================================================================
# BRAM
# =============================================================================
set datafile missing '-1'
set size 0.5,0.5
set origin 0.5,0.0 # Bottom-Right
set key left
#set xlabel "Reuse Factor" font ",8"
#set ylabel "BRAM" offset 5,0 font ",8"
plot CSV_FILE using 17:10 title "BRAM" with linespoints linestyle 1

unset multiplot
