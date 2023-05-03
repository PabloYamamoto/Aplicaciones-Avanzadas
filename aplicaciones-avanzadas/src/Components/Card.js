import React, { useState } from "react";

import "./Card.css";
import FileUpload from "./FileUpload";

const Card = () => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);

  return (
    <div
      className={
        file ? "card success-border" : error ? "card error-border" : "card"
      }
    >
      {file && <h1 className="file">{file.name.slice(0, -4)}</h1>}
      <div className="content-card">
        <FileUpload setFile={setFile} setError={setError} />
        <div className="output">
          {error && <div className="error">{error}</div>}
        </div>
      </div>
    </div>
  );
};

export default Card;
