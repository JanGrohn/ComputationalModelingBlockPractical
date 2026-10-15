import React, { useState } from "react";
import { Button, Container, Box, Typography } from "@mui/material";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import Choice from "./components/Choice";
import EnterData from "./components/EnterData"; 
import TwoButtons from "./components/TwoButtons"; 

import { Helmet } from "react-helmet";




function FullscreenMessage({ onConfirm }) {
  return (
    <Container>
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Typography variant="h4" gutterBottom>
          This study will be displayed in full screen
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<FullscreenIcon />}
          onClick={onConfirm}
        >
          Confirm
        </Button>
      </Box>
    </Container>
  );
}

window.participantData = [];

function App() {

  const [isFullScreen, setIsFullScreen] = useState(false);
  const [textSubmitted, setTextSubmitted] = useState(false);
  const [conditionSelected, setConditionSelected] = useState(false);
  const [selectedCondition, setSelectedCondition] = useState("");

  const handleTextInputSubmit = (text) => {
    setTextSubmitted(true);
    window.participantData.push({userID:text}); 
  };

  const handleConditionSelect = (selectedCondition) => {
    setConditionSelected(true);
    setSelectedCondition(selectedCondition);
    window.participantData.push({condition:selectedCondition}); 
    // console.log(selectedCondition);
  };


  const handleFullscreenConfirm = () => {
    setIsFullScreen(true);
  
    const docEl = document.documentElement;
    const requestFullScreen = docEl.requestFullscreen || docEl.mozRequestFullScreen || docEl.webkitRequestFullScreen || docEl.msRequestFullscreen;
    
    if (requestFullScreen) {
      requestFullScreen.call(docEl);
    }
  };

  return (
    <div>
      <Helmet>
        <script
          src="https://cdn.jsdelivr.net/npm/preloadjs@1.0.1/lib/preloadjs.min.js"
          crossorigin="anonymous"
          async
        ></script>
      </Helmet>
      {!isFullScreen ? (
        <FullscreenMessage onConfirm={handleFullscreenConfirm} />
      ) : !textSubmitted ? ( 
        <EnterData onSubmit={handleTextInputSubmit} />
      ) : !conditionSelected? (
        // <TwoButtons onClick={handleConditionSelect} />
        <TwoButtons onClick={handleConditionSelect} />  
      ) : (
        <Choice condition={selectedCondition} />
      )}  
    </div>
  );
}

export default App;
