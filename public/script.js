const form = document.getElementById("predictionForm");

const loading = document.getElementById("loading");
const result = document.getElementById("result");
const error = document.getElementById("error");

const recommendation = document.getElementById("recommendation");
const confidence = document.getElementById("confidence");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    result.classList.add("hidden");
    error.classList.add("hidden");

    loading.classList.remove("hidden");


    const data = {

        device_age_years:
            parseFloat(document.getElementById("device_age_years").value),

        battery_health:
            parseFloat(document.getElementById("battery_health").value),

        screen_damage:
            parseInt(document.getElementById("screen_damage").value),

        body_condition:
            parseInt(document.getElementById("body_condition").value),

        repair_cost:
            parseFloat(document.getElementById("repair_cost").value),

        device_value:
            parseFloat(document.getElementById("device_value").value),

        previous_repairs:
            parseInt(document.getElementById("previous_repairs").value),

        warranty_remaining:
            parseInt(document.getElementById("warranty_remaining").value)
    };


    try {

        const response = await fetch(
            "/api/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const resultData = await response.json();


        if (!response.ok) {
            throw new Error("Invalid input. Please check your values.");
        }


        recommendation.textContent =
            resultData.recommendation;

        confidence.textContent =
            resultData.confidence + "%";


        result.classList.remove("hidden");

    }

    catch (err) {

        error.textContent = err.message;

        error.classList.remove("hidden");

    }

    finally {

        loading.classList.add("hidden");

    }

});