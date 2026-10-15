import React from "react";
import { Button, Container, Box } from "@mui/material";

/** Condition 1 is the stable schedule, condition 2 the volatile one. */
function TwoButtons({ onClick }) {
  return (
    <Container>
      <Box
        display="flex"
        flexDirection="row"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
        sx={{ gap: 2 }}
      >
        <Button variant="contained" color="primary" onClick={() => onClick(1)}>
          Select condition 1
        </Button>
        <Button variant="contained" color="primary" onClick={() => onClick(2)}>
          Select condition 2
        </Button>
      </Box>
    </Container>
  );
}

export default TwoButtons;
