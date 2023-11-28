import './App.css';
import React, { useState } from 'react';
function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');

  const handleChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSubmit = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequence: inputText }),
      });

      const data = await response.json();
      setOutputText(data);
    } catch (error) {
      console.error('Error calling Flask API:', error);
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
        <textarea
          rows="4"
          cols="50"
          readOnly
          value={outputText}
          placeholder="Output will be displayed here"
        />
        {/* <p>{outputText}</p> */}
      </div>

    </div>
  );
}

export default App;
