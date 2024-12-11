const reviewInput = document.getElementById("review-input").value;

console.log("Sending review input:", reviewInput);  // Log the input to see what is being sent

fetch('/y_predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body:JSON.stringify({ "Sentence": reviewInput })

})
.then(response => response.json())
.then(data => {
    console.log("Response from backend:", data);  // Log the response from the backend
    document.getElementById("result").innerHTML = data.prediction_text;
})
.catch(error => {
    console.error("Error:", error);  // Log any errors that occur
    document.getElementById("result").innerHTML = "An error occurred.";
});
