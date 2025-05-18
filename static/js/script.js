// Light/Dark Mode Toggle
document.getElementById('modeToggle').addEventListener('click', function () {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    // Optionally save the theme to local storage
    localStorage.setItem('theme', newTheme);
});

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function () {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
});

// Interactive Chart for Blood Sugar Levels
function initializeBloodSugarChart() {
    const ctx = document.getElementById('bloodSugarChart');
    if (ctx) {
        const chartContext = ctx.getContext('2d');
        new Chart(chartContext, {
            type: 'line',
            data: {
                labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'], // Example data
                datasets: [{
                    label: 'Blood Sugar Levels',
                    data: [80, 90, 85, 95, 88], // Example data
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderWidth: 2,
                    fill: true,
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Blood Sugar Level (mg/dL)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Days'
                        }
                    }
                }
            }
        });
    } else {
        console.error("Canvas for blood sugar chart not found.");
    }
}

// Daily Motivation
function displayRandomQuote() {
    const quotes = [
        "Stay positive, work hard, and make it happen!",
        "Every day may not be good, but there’s something good in every day.",
        "Believe you can, and you're halfway there.",
        "You are stronger than you think!",
        "Success is not final; failure is not fatal: It is the courage to continue that counts."
    ];

    const randomIndex = Math.floor(Math.random() * quotes.length);
    document.getElementById('motivationQuote').innerText = quotes[randomIndex];
}

// Event listeners
document.addEventListener('DOMContentLoaded', function () {
    initializeBloodSugarChart();
    displayRandomQuote();
    document.getElementById('get-tip').addEventListener('click', displayRandomQuote);
});
