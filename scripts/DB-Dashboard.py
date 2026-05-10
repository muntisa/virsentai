#!/usr/bin/env python3
"""
Database Analytics Dashboard

Generates an HTML analytics dashboard for virus host distribution
using Chart.js visualizations.

Usage:
    python DB-Dashboard.py

Input:
    - db/db-viruses.sqlite3 (cleaned viruses table)

Output:
    - DB-Dashboard.html (standalone HTML dashboard)

Statistics Generated:
    - Host class distribution (Homo sapiens, Unknown, Other)
    - Sequence length statistics (min, max, mean, median, std)
    - Histogram data for all lengths
    - Histogram data for lengths ≤160K

Visualizations:
    - Pie chart: Host distribution
    - Bar chart: Top 15 non-human hosts
    - Histogram: Sequence length distribution

Notes:
    - Standalone HTML with embedded CSS and JS
    - Uses Chart.js from CDN
    - Dark theme matching project style
"""

import sqlite3
import json
import os

def get_length_histogram(cursor, host_condition, max_length=None, bins=25):
    if host_condition:
        query = f"SELECT length FROM viruses WHERE {host_condition}"
    else:
        query = "SELECT length FROM viruses"
    
    if max_length:
        query = f"SELECT length FROM viruses WHERE length <= {max_length}" + (f" AND {host_condition}" if host_condition else "")
    
    cursor.execute(query)
    lengths = [row[0] for row in cursor.fetchall() if row[0]]
    
    if not lengths:
        return [], []
    
    min_len = min(lengths)
    max_len = max(lengths)
    bin_size = (max_len - min_len) / bins if max_len > min_len else 1
    
    histogram = [0] * bins
    labels = []
    
    for i in range(bins):
        bin_start = min_len + i * bin_size
        if bin_size >= 1000:
            labels.append(f"{int(bin_start/1000)}K")
        else:
            labels.append(str(int(bin_start)))
        
        for length in lengths:
            if bin_start <= length < bin_start + bin_size:
                histogram[i] += 1
        if i == bins - 1:
            for length in lengths:
                if length == max_len:
                    histogram[i] += 1
    
    return labels, histogram

def get_stats(cursor, host_condition):
    if host_condition:
        query = f"SELECT length FROM viruses WHERE {host_condition}"
    else:
        query = "SELECT length FROM viruses"
    
    cursor.execute(query)
    lengths = [row[0] for row in cursor.fetchall() if row[0]]
    
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    
    lengths_sorted = sorted(lengths)
    n = len(lengths)
    mean = sum(lengths) / n
    median = lengths_sorted[n // 2]
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std = variance ** 0.5
    
    return {
        "count": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": int(mean),
        "median": median,
        "std": int(std)
    }

def get_stats_160k(cursor, host_condition):
    if host_condition:
        query = f"SELECT length FROM viruses WHERE length <= 160000 AND {host_condition}"
    else:
        query = "SELECT length FROM viruses WHERE length <= 160000"
    
    cursor.execute(query)
    lengths = [row[0] for row in cursor.fetchall() if row[0]]
    
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    
    lengths_sorted = sorted(lengths)
    n = len(lengths)
    mean = sum(lengths) / n
    median = lengths_sorted[n // 2]
    variance = sum((x - mean) ** 2 for x in lengths) / n
    std = variance ** 0.5
    
    return {
        "count": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": int(mean),
        "median": median,
        "std": int(std)
    }

def format_labels(labels):
    return json.dumps(labels)

def format_data(data):
    return json.dumps(data)

def main():
    db_path = SQLITE_CORR_VIRUSES_FILE
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Querying statistics...")
    
    all_stats = get_stats(cursor, None)
    human_stats = get_stats(cursor, "host = 'Homo sapiens'")
    unknown_stats = get_stats(cursor, "host = 'Unknown'")
    specific_stats = get_stats(cursor, "host != 'Homo sapiens' AND host != 'Unknown'")
    
    print(f"All: {all_stats['count']}, Human: {human_stats['count']}, Unknown: {unknown_stats['count']}, Specific: {specific_stats['count']}")
    
    # Stats for sequences up to 160K
    print("Querying 160K statistics...")
    all_160k_stats = get_stats_160k(cursor, None)
    human_160k_stats = get_stats_160k(cursor, "host = 'Homo sapiens'")
    unknown_160k_stats = get_stats_160k(cursor, "host = 'Unknown'")
    specific_160k_stats = get_stats_160k(cursor, "host != 'Homo sapiens' AND host != 'Unknown'")
    
    print(f"160K - All: {all_160k_stats['count']}, Human: {human_160k_stats['count']}, Unknown: {unknown_160k_stats['count']}, Specific: {specific_160k_stats['count']}")
    
    print("Generating histograms...")
    human_hist = get_length_histogram(cursor, "host = 'Homo sapiens'")
    unknown_hist = get_length_histogram(cursor, "host = 'Unknown'")
    specific_hist = get_length_histogram(cursor, "host != 'Homo sapiens' AND host != 'Unknown'")
    
    # 160K histograms
    human_160k_hist = get_length_histogram(cursor, "host = 'Homo sapiens'", 160000)
    unknown_160k_hist = get_length_histogram(cursor, "host = 'Unknown'", 160000)
    specific_160k_hist = get_length_histogram(cursor, "host != 'Homo sapiens' AND host != 'Unknown'", 160000)
    
    # Generate HTML
    print("Generating HTML...")
    
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Host Distribution Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; color: #fff; padding: 20px; }
        .header { text-align: center; padding: 25px 0; border-bottom: 2px solid #0f3460; margin-bottom: 25px; }
        .header h1 { font-size: 2.2rem; color: #00d9ff; text-shadow: 0 0 20px rgba(0, 217, 255, 0.5); }
        .header p { color: #888; margin-top: 8px; font-size: 1rem; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 25px; }
        .stat-card { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
        .stat-card .value { font-size: 1.8rem; font-weight: bold; color: #00d9ff; }
        .stat-card .label { color: #888; font-size: 0.75rem; margin-top: 5px; }
        .stat-card.human .value { color: #ff6b6b; }
        .stat-card.unknown .value { color: #ffd93d; }
        .stat-card.specific .value { color: #6bcb77; }
        .charts-section { margin-bottom: 35px; }
        .charts-section h2 { color: #00d9ff; margin-bottom: 18px; font-size: 1.4rem; border-left: 4px solid #00d9ff; padding-left: 12px; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
        .chart-box { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 18px; border: 1px solid rgba(255,255,255,0.1); }
        .chart-box h3 { color: #00d9ff; margin-bottom: 12px; font-size: 0.95rem; text-align: center; }
        .chart-box canvas { max-height: 220px; }
        .stats-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; justify-content: center; }
        .stats-item { background: rgba(0,0,0,0.3); padding: 6px 10px; border-radius: 6px; text-align: center; min-width: 70px; }
        .stats-item span { color: #888; font-size: 0.7rem; display: block; }
        .stats-item strong { color: #fff; font-size: 0.85rem; display: block; margin-top: 2px; }
        .full-width { grid-column: 1 / -1; }
        .footer { text-align: center; padding: 18px; color: #555; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Host Distribution Dashboard</h1>
        <p>Database: db-viruses.sqlite3 | Total Records: ''' + f"{all_stats['count']:,}" + '''</p>
    </div>

    <!-- Summary Stats -->
    <div class="stats-grid">
        <div class="stat-card human">
            <div class="value">''' + f"{human_stats['count']:,}" + '''</div>
            <div class="label">Homo sapiens</div>
        </div>
        <div class="stat-card unknown">
            <div class="value">''' + f"{unknown_stats['count']:,}" + '''</div>
            <div class="label">Unknown</div>
        </div>
        <div class="stat-card specific">
            <div class="value">''' + f"{specific_stats['count']:,}" + '''</div>
            <div class="label">Other (Non-Human)</div>
        </div>
        <div class="stat-card">
            <div class="value">''' + f"{all_stats['count']:,}" + '''</div>
            <div class="label">Total</div>
        </div>
    </div>

    <!-- All Hosts -->
    <div class="charts-section">
        <h2>Sequence Length Distribution by Host Type</h2>
        <div class="charts-grid">
            <div class="chart-box">
                <h3>Homo sapiens (Human) - ''' + f"{human_stats['count']:,}" + ''' records</h3>
                <canvas id="humanChart"></canvas>
                <div class="stats-row">
                    <div class="stats-item"><span>Min</span><strong>''' + f"{human_stats['min']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Max</span><strong>''' + f"{human_stats['max']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Mean</span><strong>''' + f"{human_stats['mean']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Median</span><strong>''' + f"{human_stats['median']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Std</span><strong>''' + f"{human_stats['std']:,}" + '''</strong></div>
                </div>
            </div>
            <div class="chart-box">
                <h3>Unknown - ''' + f"{unknown_stats['count']:,}" + ''' records</h3>
                <canvas id="unknownChart"></canvas>
                <div class="stats-row">
                    <div class="stats-item"><span>Min</span><strong>''' + f"{unknown_stats['min']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Max</span><strong>''' + f"{unknown_stats['max']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Mean</span><strong>''' + f"{unknown_stats['mean']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Median</span><strong>''' + f"{unknown_stats['median']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Std</span><strong>''' + f"{unknown_stats['std']:,}" + '''</strong></div>
                </div>
            </div>
            <div class="chart-box">
                <h3>Other (Non-Human) - ''' + f"{specific_stats['count']:,}" + ''' records</h3>
                <canvas id="specificChart"></canvas>
                <div class="stats-row">
                    <div class="stats-item"><span>Min</span><strong>''' + f"{specific_stats['min']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Max</span><strong>''' + f"{specific_stats['max']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Mean</span><strong>''' + f"{specific_stats['mean']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Median</span><strong>''' + f"{specific_stats['median']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Std</span><strong>''' + f"{specific_stats['std']:,}" + '''</strong></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Comparison -->
    <div class="charts-section">
        <h2>Comparison - Mean vs Median by Host Type</h2>
        <div class="charts-grid">
            <div class="chart-box full-width">
                <canvas id="compareChart"></canvas>
            </div>
        </div>
    </div>

    <!-- 160K Section -->
    <div class="charts-section">
        <h2>Sequence Length Distribution (≤160K bases) by Host Type</h2>
        <div class="charts-grid">
            <div class="chart-box">
                <h3>Homo sapiens (≤160K) - ''' + f"{human_160k_stats['count']:,}" + ''' records</h3>
                <canvas id="human160kChart"></canvas>
                <div class="stats-row">
                    <div class="stats-item"><span>Min</span><strong>''' + f"{human_160k_stats['min']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Max</span><strong>''' + f"{human_160k_stats['max']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Mean</span><strong>''' + f"{human_160k_stats['mean']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Median</span><strong>''' + f"{human_160k_stats['median']:,}" + '''</strong></div>
                </div>
            </div>
            <div class="chart-box">
                <h3>Unknown (≤160K) - ''' + f"{unknown_160k_stats['count']:,}" + ''' records</h3>
                <canvas id="unknown160kChart"></canvas>
                <div class="stats-row">
                    <div class="stats-item"><span>Min</span><strong>''' + f"{unknown_160k_stats['min']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Max</span><strong>''' + f"{unknown_160k_stats['max']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Mean</span><strong>''' + f"{unknown_160k_stats['mean']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Median</span><strong>''' + f"{unknown_160k_stats['median']:,}" + '''</strong></div>
                </div>
            </div>
            <div class="chart-box">
                <h3>Other Non-Human (≤160K) - ''' + f"{specific_160k_stats['count']:,}" + ''' records</h3>
                <canvas id="specific160kChart"></canvas>
                <div class="stats-row">
                    <div class="stats-item"><span>Min</span><strong>''' + f"{specific_160k_stats['min']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Max</span><strong>''' + f"{specific_160k_stats['max']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Mean</span><strong>''' + f"{specific_160k_stats['mean']:,}" + '''</strong></div>
                    <div class="stats-item"><span>Median</span><strong>''' + f"{specific_160k_stats['median']:,}" + '''</strong></div>
                </div>
            </div>
        </div>
    </div>

    <!-- 160K Comparison -->
    <div class="charts-section">
        <h2>Comparison (≤160K) - Mean vs Median by Host Type</h2>
        <div class="charts-grid">
            <div class="chart-box full-width">
                <canvas id="compare160kChart"></canvas>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Generated from db-viruses.sqlite3 | Host Distribution Dashboard</p>
    </div>

    <script>
        const humanData = {
            labels: ''' + format_labels(human_hist[0]) + ''',
            data: ''' + format_data(human_hist[1]) + '''
        };
        const unknownData = {
            labels: ''' + format_labels(unknown_hist[0]) + ''',
            data: ''' + format_data(unknown_hist[1]) + '''
        };
        const specificData = {
            labels: ''' + format_labels(specific_hist[0]) + ''',
            data: ''' + format_data(specific_hist[1]) + '''
        };

        // 160K data
        const human160kData = {
            labels: ''' + format_labels(human_160k_hist[0]) + ''',
            data: ''' + format_data(human_160k_hist[1]) + '''
        };
        const unknown160kData = {
            labels: ''' + format_labels(unknown_160k_hist[0]) + ''',
            data: ''' + format_data(unknown_160k_hist[1]) + '''
        };
        const specific160kData = {
            labels: ''' + format_labels(specific_160k_hist[0]) + ''',
            data: ''' + format_data(specific_160k_hist[1]) + '''
        };

        const colors = {
            human: { bg: "rgba(255,107,107,0.7)", border: "#ff6b6b" },
            unknown: { bg: "rgba(255,217,61,0.7)", border: "#ffd93d" },
            specific: { bg: "rgba(107,203,119,0.7)", border: "#6bcb77" }
        };

        function createChart(canvasId, data, color) {
            const ctx = document.getElementById(canvasId).getContext("2d");
            return new Chart(ctx, {
                type: "bar",
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data.data,
                        backgroundColor: color.bg,
                        borderColor: color.border,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.1)" }, ticks: { color: "#888", maxTicksLimit: 8 } },
                        x: { grid: { display: false }, ticks: { color: "#888", maxRotation: 45, font: { size: 9 } } }
                    }
                }
            });
        }

        createChart("humanChart", humanData, colors.human);
        createChart("unknownChart", unknownData, colors.unknown);
        createChart("specificChart", specificData, colors.specific);

        // 160K charts
        createChart("human160kChart", human160kData, colors.human);
        createChart("unknown160kChart", unknown160kData, colors.unknown);
        createChart("specific160kChart", specific160kData, colors.specific);

        // Comparison chart
        const compareCtx = document.getElementById("compareChart").getContext("2d");
        new Chart(compareCtx, {
            type: "bar",
            data: {
                labels: ["Homo sapiens", "Unknown", "Other Non-Human"],
                datasets: [
                    { label: "Mean", data: [''' + str(human_stats['mean']) + ', ' + str(unknown_stats['mean']) + ', ' + str(specific_stats['mean']) + '''], backgroundColor: "rgba(0,217,255,0.7)", borderColor: "#00d9ff", borderWidth: 2 },
                    { label: "Median", data: [''' + str(human_stats['median']) + ', ' + str(unknown_stats['median']) + ', ' + str(specific_stats['median']) + '''], backgroundColor: "rgba(155,89,182,0.7)", borderColor: "#9b59b6", borderWidth: 2 }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "top", labels: { color: "#fff" } } },
                scales: {
                    y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.1)" }, ticks: { color: "#888" } },
                    x: { grid: { display: false }, ticks: { color: "#888" } }
                }
            }
        });

        // 160K Comparison chart
        const compare160kCtx = document.getElementById("compare160kChart").getContext("2d");
        new Chart(compare160kCtx, {
            type: "bar",
            data: {
                labels: ["Homo sapiens", "Unknown", "Other Non-Human"],
                datasets: [
                    { label: "Mean", data: [''' + str(human_160k_stats['mean']) + ', ' + str(unknown_160k_stats['mean']) + ', ' + str(specific_160k_stats['mean']) + '''], backgroundColor: "rgba(0,217,255,0.7)", borderColor: "#00d9ff", borderWidth: 2 },
                    { label: "Median", data: [''' + str(human_160k_stats['median']) + ', ' + str(unknown_160k_stats['median']) + ', ' + str(specific_160k_stats['median']) + '''], backgroundColor: "rgba(155,89,182,0.7)", borderColor: "#9b59b6", borderWidth: 2 }
                ]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "top", labels: { color: "#fff" } } },
                scales: {
                    y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.1)" }, ticks: { color: "#888" } },
                    x: { grid: { display: false }, ticks: { color: "#888" } }
                }
            }
        });
    </script>
</body>
</html>'''
    
    # Save to file
    output_path = 'DB-Dashboard.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Dashboard saved to {output_path}")
    
    conn.close()

if __name__ == '__main__':
    main()