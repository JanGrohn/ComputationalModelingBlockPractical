import React, { useState } from "react";
import { Button, Container, Box, Typography } from "@mui/material";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import Choice from "./components/Choice";
import EnterData from "./components/EnterData";
import TwoButtons from "./components/TwoButtons";

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
    window.participantData.push({ userID: text });
  };

  const handleConditionSelect = (condition) => {
    setConditionSelected(true);
    setSelectedCondition(condition);
    window.participantData.push({ condition });
  };

  const handleFullscreenConfirm = () => {
    setIsFullScreen(true);
    const docEl = document.documentElement;
    const requestFullScreen =
      docEl.requestFullscreen ||
      docEl.mozRequestFullScreen ||
      docEl.webkitRequestFullScreen ||
      docEl.msRequestFullscreen;
    if (requestFullScreen) {
      requestFullScreen.call(docEl);
    }
  };

  if (!isFullScreen) return <FullscreenMessage onConfirm={handleFullscreenConfirm} />;
  if (!textSubmitted) return <EnterData onSubmit={handleTextInputSubmit} />;
  if (!conditionSelected) return <TwoButtons onClick={handleConditionSelect} />;
  return <Choice condition={selectedCondition} />;
}

export default App;
