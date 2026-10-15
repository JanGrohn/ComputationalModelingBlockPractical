import React, { useState } from "react";
import { Button, Container, Box, Typography, TextField } from "@mui/material";

function EndStudy({ url }) {
  const [text, setText] = useState("");

  const handleDownload = () => {
    window.participantData.push({
      comments: text,
    });
    const data = JSON.stringify(window.participantData);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "participantData.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Container>
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Typography variant="h6" gutterBottom>
          Thanks for participating! Do you have any comments about the study?
        </Typography>
        <TextField
          label="Comments"
          variant="outlined"
          value={text}
          onChange={(event) => setText(event.target.value)}
          multiline
          rows={4}
          sx={{
            mb: 2,
            width: "80%",
          }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleDownload}
          sx={{
            mt: 2,
          }}
        >
          Download results
        </Button>
      </Box>
    </Container>
  );
}

export default EndStudy;