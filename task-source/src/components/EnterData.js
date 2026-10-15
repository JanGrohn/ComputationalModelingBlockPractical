import React, { useState } from "react";
import { Button, Container, Box, Typography, TextField } from "@mui/material";

function EnterData({ onSubmit }) {
  const [text, setText] = useState("");

  return (
    <Container>
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        {/* <Typography variant="h4" gutterBottom>
          Please enter text
        </Typography> */}
        <TextField
          label="Enter a username"
        //   variant="outlined"
        //   value={text}
          onChange={(event) => setText(event.target.value)}
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={() => onSubmit(text)}
          disabled={!text}
        >
          Confirm
        </Button>
      </Box>
    </Container>
  );
}

export default EnterData;