import './App.css';
import React, { useState } from 'react';
import { Button } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import GradientButton from 'react-linear-gradient-button';
import ReactFlipCard from 'reactjs-flip-card';

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState([]);
  const [inputArray, setInputArray] = useState([]);
  const [finalOutputText, setFinalOutputText] = useState([]);
  const [showMetrics, setshowMetrics] = useState();
  const styles = {
    card: { background: 'rgb(255, 255, 220)', color: 'black', borderRadius: 30, fontSize:"18px" },
    centeredContent: {display: 'flex', justifyContent: 'center', alignItems: 'center', }
  }

  const handleChange = (event) => {
    setInputText(event.target.value);
  };

  const handleFinalOuput = () => {
    let finalOutput = [];
    for (let i = 0; i <= inputArray.length - 1; i++) {
      let tempString = inputArray[i].replace(/\[MASK\]/g, "\n[" + outputText[i] + "]\n");
      finalOutput[i] = tempString;
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

  const handleDoNotShowMetrics = () => {
    setshowMetrics(false);
  }

  const handleShowMetrics = () => {
    setshowMetrics(true);
  }

  return (
    <div className="App">
      <h1 style={{ color: "white" }}>GENOME SEQUENCE PREDICTION</h1>
      <br />
      <div style={{ display: 'flex' }}>
        <GradientButton
          style={{ background: "rgb(40, 91, 82)", width: "50%"}}
          onClick={handleDoNotShowMetrics}>TEST MODEL</GradientButton>

        <GradientButton
          style={{ background: "rgb(40, 91, 82)", width: "50%", marginLeft: "30px" }}
          onClick={handleShowMetrics}>MODEL METRICS</GradientButton>
      </div>

      {showMetrics == false ? (
        <div>
          <br/>
          <h2 style={{ width: "fit-content", color: "white" }}>INPUT:</h2>
          <textarea
            rows="7"
            cols="50"
            value={inputText}
            onChange={handleChange}
            placeholder="Enter Masked Sequence"
          />
          <br /><br />
          {/* <Button variant="success" onClick={handleSubmit}>SUBMIT</Button> */}
          <button className="Success" onClick={handleSubmit} style={{ padding: "10px", fontSize:"20px" }}>SUBMIT</button>
          <br />
          <div>
            <h2 style={{ width: "fit-content", color: "white" }}>RECONSTRUCTED SEQUENCE:</h2>
            {finalOutputText.length > 0 ? (
              <div>
                {finalOutputText.map((output, index) => (
                  <textarea
                    key={index}
                    rows="5"
                    cols="50"
                    readOnly
                    value={output}
                    placeholder="Output will be displayed here"
                  />
                ))}
              </div>
            ) : (
              ""
            )}
          </div>
        </div>
      ) : (
        ""
      )}

      {showMetrics == true ? (
        <>
        <div style={{ display: "flex" , height:"250px"}}>
        <div style={{margin:"10px", width:"33.33%"}}>
            <ReactFlipCard
              frontStyle={{...styles.card, ...styles.centeredContent}}
              backStyle={{ ...styles.card,...styles.centeredContent}}
              frontComponent={<div><strong>Tokenizer CheckPoint:</strong> Default(zhihan1996/DNABERT-2-117M) <br/>
              <strong>Model CheckPoint:</strong> Default(zhihan1996/DNABERT-2-117M)</div>}
              backComponent={<div><strong>Avg overlap accuraccy:</strong> 39.0823%</div>}
              containerStyle={{width: "100%", height: "100%"}}
            />
          </div>
          <div style={{margin:"10px", width:"33.33%"}}>
            <ReactFlipCard
              frontStyle={{...styles.card, ...styles.centeredContent}}
              backStyle={{ ...styles.card,...styles.centeredContent}}
              frontComponent={<div><strong>Tokenizer CheckPoint:</strong> Default(zhihan1996/DNABERT-2-117M) <br/>
              <strong>Model CheckPoint:</strong> Trained on 5% of Uniprot Data</div>}
              backComponent={<div><strong>Avg. Overlap Accuraccy: </strong>43.3635%</div>}
              containerStyle={{width: "100%", height: "100%"}}
            />
          </div>
          <div style={{margin:"10px", width:"33.33%"}}>
            <ReactFlipCard
             frontStyle={{...styles.card, ...styles.centeredContent}}
             backStyle={{ ...styles.card,...styles.centeredContent}}
              frontComponent={<div><strong>Tokenizer Checkpoint:</strong> Trained on SwissProt Trainset <br/>
                <strong>Model Checkpoint:</strong> Halfway trained on combination of Swissprot and Uniprot Data</div>}
              backComponent={<div><strong>Avg Overlap Accuraccy:</strong> 37.7311%</div>}
              containerStyle={{width: "100%", height: "100%"}}
            />
          </div>
        </div>

        <div style={{ display: "flex" , height:"250px"}}>
          <div style={{margin:"10px", width:"33.33%"}}>
            <ReactFlipCard
              frontStyle={{...styles.card, ...styles.centeredContent}}
              backStyle={{ ...styles.card,...styles.centeredContent}}
              frontComponent={<div><strong>Tokenizer Checkpoint:</strong> Default(zhihan1996/DNABERT-2-117M) <br/>
              <strong>Model Checkpoint:</strong> Halfway trained on combination of Swissprot and Uniprot Data</div>}
              backComponent={<div><strong>Avg Overlap Accuraccy:</strong> 40.3355%</div>}
              containerStyle={{width: "100%", height: "100%"}}
            />
          </div>
          <div style={{margin:"10px", width:"33.33%"}}>
            <ReactFlipCard
              frontStyle={{...styles.card, ...styles.centeredContent}}
              backStyle={{ ...styles.card,...styles.centeredContent}}
              frontComponent={<div><strong>Tokenizer Checkpoint:</strong> Default(zhihan1996/DNABERT-2-117M) <br/>
              <strong>Model Checkpoint:</strong> Fully trained on combination of Swissprot and Uniprot Data</div>}
              backComponent={<div><strong>Avg Overlap Accuraccy:</strong> </div>}
              containerStyle={{width: "100%", height: "100%"}}
            />
          </div>
          <div style={{margin:"10px", width:"33.33%"}}>
            <ReactFlipCard
              frontStyle={{...styles.card, ...styles.centeredContent}}
              backStyle={{ ...styles.card,...styles.centeredContent}}
              frontComponent={<div><strong>Tokenizer Checkpoint:</strong> Trained on SwissProt Trainset <br/>
                <strong>Model Checkpoint:</strong> Fully trained on combination of Swissprot and Uniprot Data</div>}
              backComponent={<div><strong>Avg Overlap Accuraccy:</strong> </div>}
              containerStyle={{width: "100%", height: "100%"}}
            />
          </div>
        </div>
        </>
      ) : ""}

    </div>
  );
}

export default App;

