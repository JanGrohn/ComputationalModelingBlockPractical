import React, { useState } from "react";
import { Box, Paper, Button, Typography, Slide, Stack, Alert, AlertTitle, Collapse } from "@mui/material";
import { styled } from "@mui/system";
import TextTransition, { presets } from "react-text-transition";
import CheckIcon from "@mui/icons-material/Check";
import ForwardIcon from "@mui/icons-material/Forward";
import CraneImage from "./img/bird-crane-shape.svg";
import DeerImage from "./img/deer-silhouette.svg";
import EndStudy from "./EndStudy";
import { buildSchedule, N_TRIALS } from "../schedule";

const PaperWrapper = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    backgroundColor: "white",
    borderRadius: "12px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    flexGrow: 1,
    flexBasis: 0,
    maxHeight: "50vh",
    maxWidth: "50vw",
}));

function Choice({ condition }) {
    const [schedule] = useState(() => buildSchedule(condition));
    const { reward1, reward2, displaySide, oneRewarded } = schedule;

    const [selectedPaper, setSelectedPaper] = useState(null);
    const [confirmDisabled, setConfirmDisabled] = useState(true);
    const [newPaperAnimation, setNewPaperAnimation] = useState(true);
    const [currentIndex, setCurrentIndex] = useState(0);

    if (currentIndex >= N_TRIALS) {
        return <EndStudy />;
    }

    const rewardedThisTrial = oneRewarded[currentIndex];
    const leftIsOptionOne = Boolean(displaySide[currentIndex]);

    const handlePaperClick = (paper) => {
        window.participantData.push({
            trial: currentIndex,
            time: new Date().getTime(),
            type: paper === "left" ? "Select Left" : "Select Right",
            displaySide: displaySide[currentIndex],
            reward1: reward1[currentIndex],
            reward2: reward2[currentIndex],
            oneRewarded: rewardedThisTrial,
        });
        setSelectedPaper(paper);
        setTimeout(() => setConfirmDisabled(false), 1000);
    };

    const handleConfirmClick = () => {
        window.participantData.push({
            trial: currentIndex + 1,
            time: new Date().getTime(),
            type: "Trial started",
        });
        setConfirmDisabled(true);
        setNewPaperAnimation(false);
        setTimeout(() => {
            setSelectedPaper(null);
            setNewPaperAnimation(true);
            setCurrentIndex((index) => index + 1);
        }, 1000);
    };

    // what each side shows: its magnitude before a choice is made, and after
    // one, the magnitude if that side was rewarded or a dash if it was not
    const sideValue = (sideIsOptionOne) => {
        const magnitude = sideIsOptionOne ? reward1[currentIndex] : reward2[currentIndex];
        if (!selectedPaper) return magnitude;
        return sideIsOptionOne === rewardedThisTrial ? magnitude : "-";
    };

    const card = (side, sideIsOptionOne, direction) => (
        <Slide direction={direction} in={newPaperAnimation}>
            <PaperWrapper
                elevation={3}
                sx={{ backgroundColor: selectedPaper === side ? "lightblue" : "white" }}
            >
                <div style={{ userSelect: "none", display: "flex", justifyContent: "center", alignItems: "center", height: "100%", flexGrow: 1, overflow: "hidden" }}>
                    <img
                        src={sideIsOptionOne ? DeerImage : CraneImage}
                        alt={sideIsOptionOne ? "the green option" : "the orange option"}
                        style={{
                            maxWidth: "100%",
                            maxHeight: "26vh",
                            objectFit: "contain",
                            filter: "drop-shadow(3px 0px black)",
                        }}
                    />
                </div>
                <Typography variant="h2" mt={2} sx={{ userSelect: "none", fontSize: "5rem", fontWeight: 500 }}>
                    <TextTransition springConfig={presets.gentle} inline>
                        {sideValue(sideIsOptionOne)}
                    </TextTransition>
                </Typography>
                <Box mt={2}>
                    <Button
                        variant="contained"
                        color="primary"
                        disabled={selectedPaper !== null}
                        onClick={() => handlePaperClick(side)}
                        startIcon={<CheckIcon />}
                    >
                        Select
                    </Button>
                </Box>
            </PaperWrapper>
        </Slide>
    );

    return (
        <Box
            sx={{
                backgroundColor: "white",
                minHeight: "100vh",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
                width: { xs: 5 / 10, sm: 5 / 10 },
                position: "absolute",
                left: "50%",
                top: "50%",
                transform: "translate(-50%, -50%)",
            }}
        >
            <Box
                style={{ overflow: "hidden" }}
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                    minHeight: "10vh",
                    backgroundColor: "#f5f5f5",
                    borderRadius: "8px",
                    boxShadow: "0px 0px 16px rgba(0, 0, 0, 0.12)",
                    padding: "2rem",
                }}
            >
                <Collapse in={currentIndex === 0}>
                    <Alert icon={false} severity="info" sx={{ fontSize: "h7", mb: 2 }}>
                        <AlertTitle sx={{ fontSize: "h6" }}><strong>Instructions</strong></AlertTitle>
                        <strong>You are repeatedly choosing between two pictures. You have to select
                        which picture you want to choose by clicking the corresponding button. The
                        point amount you can win from selecting a picture is shown below it. Choose
                        one of the pictures now.</strong>
                    </Alert>
                </Collapse>
                <Collapse in={currentIndex === 1}>
                    <Alert icon={false} severity="warning" sx={{ fontSize: "h7", mb: 2 }}>
                        <AlertTitle sx={{ fontSize: "h6" }}><strong>Instructions</strong></AlertTitle>
                        <strong>As you saw, only one of the pictures actually yields points after
                        your choice. The overall points you won from the pictures you choose
                        directly translate into a bonus payment at the end of the study. It is your
                        task to figure out which picture will be rewarded next. Choose one of the
                        pictures again now.</strong>
                    </Alert>
                </Collapse>
                <Collapse in={currentIndex === 2}>
                    <Alert icon={false} severity="success" sx={{ fontSize: "h7", mb: 2 }}>
                        <AlertTitle sx={{ fontSize: "h6" }}><strong>Instructions</strong></AlertTitle>
                        <strong>Try to get as many points as possible to receive a larger bonus
                        payment. Continue choosing between the pictures. The study starts now.</strong>
                    </Alert>
                </Collapse>

                <Stack direction="row" justifyContent="space-around" width="100%" spacing={4}>
                    {card("left", leftIsOptionOne, "right")}
                    {card("right", !leftIsOptionOne, "left")}
                </Stack>

                <Box mt={4}>
                    <Button
                        variant="contained"
                        color="secondary"
                        disabled={confirmDisabled}
                        onClick={handleConfirmClick}
                        sx={{
                            borderRadius: "12px",
                            backgroundColor: confirmDisabled ? "grey" : "secondary.main",
                        }}
                        startIcon={<ForwardIcon />}
                    >
                        Next Choice
                    </Button>
                </Box>
            </Box>
        </Box>
    );
}

export default Choice;
