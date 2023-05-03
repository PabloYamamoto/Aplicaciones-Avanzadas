import "./MainContent.css";
import Card from "./Card";

const MainContent = () => {
  return (
    <div className="main-content">
      <Card />
      <div className="vs-container">
        <p className="vs">VS.</p>
      </div>
      <Card />
    </div>
  );
};

export default MainContent;
