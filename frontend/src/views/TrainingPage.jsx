import React, { useState } from "react";
import axios from "axios";
import { Box, Stack, Flex, Input, Button, Text } from "@chakra-ui/react";
import TrainingTasks from "views/admin/dataTables/components/TrainingTasks";
import PastTrainingTasks from "views/admin/dataTables/components/PastTrainingTasks";
import { useSelector } from "react-redux";

export default function Training() {
  const [selectedTask, setSelectedTask] = useState(null);
  const wallet_id = useSelector((state) => state.person.wallet_id);
  const [tokens, setTokens] = useState("");

  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }} h="100vh" overflow="auto">
      <Stack spacing="20px">
        
        {/* Current Training Tasks Table */}
          <TrainingTasks onTaskSelect={setSelectedTask} />
          
        

        {/* Past Training Tasks Section */}
        <Box>
          <PastTrainingTasks />
        </Box>
      </Stack>
    </Box>
  );
}
