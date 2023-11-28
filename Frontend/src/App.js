import './App.css';
import React, { useState } from 'react';
// Existing imports...

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState([]);
  const [inputArray, setInputArray] = useState([]);
  const [finalOutputText, setFinalOutputText] = useState([]);

  const handleChange = (event) => {
    setInputText(event.target.value);
  };

  const handleFinalOuput = () => {
    let finalOutput = [];
    for(let i=0;i<=inputArray.length-1;i++){
      let tempString = inputArray[i].replace(/\[MASK\]/g, "\n[" + outputText[i] + "]\n");
      finalOutput[i]=tempString;
    }
    setFinalOutputText(finalOutput);
  }

  const handleSubmit = async () => {
    let trimmedString = inputText.trim();
    let valuesArray = trimmedString.split(',');
    let trimmedValuesArray = valuesArray.map((value) => value.trim());
    setInputArray(trimmedValuesArray);
    for (let i = 0; i <= trimmedValuesArray.length - 1; i++) {
      try {
        const response = await fetch('http://127.0.0.1:5000/api', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ sequence: trimmedValuesArray[i] }),
        });

        const data = await response.json();
        setOutputText((prevOutputText) => [...prevOutputText, data]);
        handleFinalOuput();
      } catch (error) {
        console.error('Error calling Flask API:', error);
      }
    }
  };

  return (
    <div className="App">
      <h1>GENOME SEQUENCE PREDICTION</h1>
      <textarea
        rows="4"
        cols="50"
        value={inputText}
        onChange={handleChange}
        placeholder="Enter Masked Sequence"
      />
      <br />
      <button onClick={handleSubmit}>SUBMIT</button>
      <br />
      <div>
        <h2>OUTPUT:</h2>
        {finalOutputText.length > 0 ? (
          finalOutputText.map((output, index) => (
            <textarea
              key={index}
              rows="4"
              cols="50"
              readOnly
              value={output}
              placeholder="Output will be displayed here"
            />
          ))
        ) : (
          <p>No output yet</p>
        )}
      </div>
    </div>
  );
}

export default App;

