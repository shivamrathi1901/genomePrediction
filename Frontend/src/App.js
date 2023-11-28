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
      const response = await fetch('http://localhost:5000/api', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: inputText }),
      });

      const data = await response.json();
      setOutputText(data.output);
    } catch (error) {
      console.error('Error calling Flask API:', error);
    }
  };

  return (
    <div className="App">
      <h1>React Flask App</h1>
      <textarea
        rows="4"
        cols="50"
        value={inputText}
        onChange={handleChange}
        placeholder="Enter 128 characters"
      />
      <br />
      <button onClick={handleSubmit}>Submit</button>
      <br />
      <div>
        <h2>Output:</h2>
        <p>{outputText}</p>
      </div>

    </div>
  );
}

export default App;
