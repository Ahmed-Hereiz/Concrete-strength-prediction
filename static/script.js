// static/script.js - Frontend JavaScript

document.addEventListener('DOMContentLoaded', () => {
    const shortNames = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age'];
    const units = ['kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'kg/m³', 'days'];
    const ranges = [
        [90.0, 600.0, 300.0],
        [0.0, 400.0, 100.0],
        [0.0, 220.0, 50.0],
        [110.0, 270.0, 180.0],
        [0.0, 35.0, 5.0],
        [750.0, 1200.0, 1000.0],
        [550.0, 1050.0, 800.0],
        [1.0, 400.0, 28.0]
    ];

    const slidersContainer = document.querySelector('.sliders-container');
    const sliders = [];

    // Generate sliders
    ranges.forEach((range, index) => {
        const [min, max, defaultVal] = range;
        const widget = document.createElement('div');
        widget.className = 'slider-widget';

        widget.innerHTML = `
            <div class="slider-label">
                <span>${shortNames[index]}</span>
                <span class="slider-value">${defaultVal.toFixed(1)} ${units[index]}</span>
            </div>
            <input type="range" min="${min}" max="${max}" value="${defaultVal}" step="0.1">
            <div class="range-labels">
                <span>${min}</span>
                <span>${max}</span>
            </div>
        `;

        const slider = widget.querySelector('input');
        const valueDisplay = widget.querySelector('.slider-value');

        slider.addEventListener('input', () => {
            valueDisplay.textContent = `${parseFloat(slider.value).toFixed(1)} ${units[index]}`;
        });

        sliders.push(slider);
        slidersContainer.appendChild(widget);
    });

    const modelSelect = document.getElementById('model-select');
    const strengthValue = document.getElementById('strength-value');
    const strengthBar = document.getElementById('strength-bar');
    const originalTab = document.getElementById('original-tab');
    const engineeredTab = document.getElementById('engineered-tab');
    const tabButtons = document.querySelectorAll('.tab-button');
    const calculateBtn = document.getElementById('calculate-btn');
    const resetBtn = document.getElementById('reset-btn');

    // Tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            button.classList.add('active');
            document.getElementById(`${button.dataset.tab}-tab`).classList.add('active');
        });
    });

    // Reset function
    resetBtn.addEventListener('click', () => {
        sliders.forEach((slider, index) => {
            const defaultVal = ranges[index][2];
            slider.value = defaultVal;
            const valueDisplay = slider.parentElement.querySelector('.slider-value');
            valueDisplay.textContent = `${defaultVal.toFixed(1)} ${units[index]}`;
        });
        calculate();
    });

    // Calculate function
    calculateBtn.addEventListener('click', calculate);
    modelSelect.addEventListener('change', calculate);

    async function calculate() {
        const inputs = sliders.map(slider => parseFloat(slider.value));
        const modelName = modelSelect.value;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ inputs, model_name: modelName })
            });

            if (!response.ok) throw new Error('Prediction failed');

            const data = await response.json();

            // Update strength
            strengthValue.textContent = data.prediction.toFixed(1);
            strengthBar.value = Math.min(100, Math.max(0, Math.round(data.prediction)));

            // Update original features
            originalTab.innerHTML = '';
            Object.entries(data.original_features).forEach(([name, value]) => {
                const row = document.createElement('div');
                row.className = 'feature-row';
                row.innerHTML = `
                    <span class="feature-name">${name}</span>
                    <span class="feature-value">${value.toFixed(1)}</span>
                `;
                originalTab.appendChild(row);
            });

            // Update engineered features
            engineeredTab.innerHTML = '';
            Object.entries(data.engineered_features).forEach(([name, value]) => {
                const row = document.createElement('div');
                row.className = 'feature-row engineered';
                row.innerHTML = `
                    <span class="feature-name">${name}</span>
                    <span class="feature-value">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                `;
                engineeredTab.appendChild(row);
            });
        } catch (error) {
            console.error(error);
            strengthValue.textContent = 'Error';
        }
    }

    // Initial calculation
    setTimeout(calculate, 100);
});