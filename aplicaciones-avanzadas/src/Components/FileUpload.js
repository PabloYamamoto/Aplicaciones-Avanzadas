import React, { useState } from "react";

const FileUpload = ({setFile, setError}) => {
    
    const types = ["application/pdf"];
    
    const handleChange = (e) => {
        let selected = e.target.files[0];
        if (selected && types.includes(selected.type)) {
        setFile(selected);
        setError(null);
        } else {
        setFile(null);
        setError("Por favor selecciona un archivo PDF");
        }
    };
    
    return (
        <form>
        <input type="file" onChange={handleChange} />
        </form>
    );
}
    
export default FileUpload;