/* eslint-disable */
import {
  Box,
  Flex,
  Icon,
  Progress,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useColorModeValue,
  Collapse,
  Input,
  Button,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  useDisclosure,
  FormControl,
  FormLabel,
} from '@chakra-ui/react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { MdCancel, MdCheckCircle, MdOutlineError, MdExpandMore, MdExpandLess } from 'react-icons/md';
import Card from 'components/card/Card';
import Menu from 'components/menu/MainMenu';
import axios from 'axios';
import * as React from 'react';
import { useSelector } from 'react-redux';

const columnHelper = createColumnHelper();

export default function ComplexTable(props) {
  const { tableData } = props;
  const [sorting, setSorting] = React.useState([]);
  const [expandedRows, setExpandedRows] = React.useState({});
  const [tokensPerTask, setTokensPerTask] = React.useState({});
  const [selectedTask, setSelectedTask] = React.useState(null);
  const [flockApiKey, setFlockApiKey] = React.useState("");
  const [hfToken, setHfToken] = React.useState("");
  const [hfUsername, setHfUsername] = React.useState("");

  const { isOpen, onOpen, onClose } = useDisclosure();
  const wallet_id = useSelector((state) => state.person.wallet_id);
  const textColor = useColorModeValue('secondaryGray.900', 'white');
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.100');

  const toggleRowExpansion = (row) => {
    setExpandedRows((prevState) => ({
      ...prevState,
      [row.id]: !prevState[row.id],
    }));
    setSelectedTask(row.original);
  };

  const handleTokensChange = (taskId, value) => {
    setTokensPerTask((prevTokens) => ({
      ...prevTokens,
      [taskId]: value,
    }));
  };

  const columns = [
    columnHelper.accessor('task-id', {
      id: 'task-id',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          TASK ID
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Text color={textColor} fontSize="sm" fontWeight="700">
            {info.getValue()}
          </Text>
        </Flex>
      ),
    }),
    columnHelper.accessor('task-name', {
      id: 'task-name',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          TASK NAME
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Text color={textColor} fontSize="sm" fontWeight="700">
            {info.getValue()}
          </Text>
        </Flex>
      ),
    }),
    columnHelper.accessor('task-status', {
      id: 'task-status',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          TASK STATUS
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Icon
            w="24px"
            h="24px"
            me="5px"
            color={
              info.getValue() === 'Finalized'
                ? 'green.500'
                : info.getValue() === 'Disable'
                ? 'red.500'
                : info.getValue() === 'Submission'
                ? 'orange.500'
                : null
            }
            as={
              info.getValue() === 'Finalized'
                ? MdCheckCircle
                : info.getValue() === 'Disable'
                ? MdCancel
                : info.getValue() === 'Submission'
                ? MdOutlineError
                : null
            }
          />
          <Text color={textColor} fontSize="sm" fontWeight="700">
            {info.getValue()}
          </Text>
        </Flex>
      ),
    }),
    columnHelper.accessor('date', {
      id: 'date',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          DATE
        </Text>
      ),
      cell: (info) => (
        <Text color={textColor} fontSize="sm" fontWeight="700">
          {info.getValue()}
        </Text>
      ),
    }),
    columnHelper.accessor('progress', {
      id: 'progress',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          PROGRESS
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Progress
            variant="table"
            colorScheme="brandScheme"
            h="8px"
            w="108px"
            value={info.getValue()}
          />
        </Flex>
      ),
    }),
  ];

  const table = useReactTable({
    data: tableData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <Card flexDirection="column" w="100%" px="0px" overflowX="auto">
      <Flex px="25px" mb="8px" justifyContent="space-between" align="center">
        <Text color={textColor} fontSize="22px" fontWeight="700">Current Tasks</Text>
        <Menu />
      </Flex>
      <Box>
        <Table variant="simple" color="gray.500" mb="24px" mt="12px">
          <Thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <Tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <Th key={header.id} borderColor={borderColor}>
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </Th>
                ))}
              </Tr>
            ))}
          </Thead>
          <Tbody>
            {table.getRowModel().rows.map((row) => (
              <React.Fragment key={row.id}>
                <Tr cursor="pointer" onClick={() => toggleRowExpansion(row)} _hover={{ bg: 'gray.100' }}>
                  {row.getVisibleCells().map((cell) => (
                    <Td key={cell.id} borderColor="transparent">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </Td>
                  ))}
                  <Td>
                    <Button
                      colorScheme="brandScheme"
                      bg="brand.500"
                      color="white"
                      onClick={() => {
                        setSelectedTask(row.original);
                        onOpen();
                      }}
                    >
                      Train
                    </Button>
                  </Td>
                </Tr>
                <Tr>
                  <Td colSpan={columns.length + 1} p="0">
                    <Collapse in={expandedRows[row.id]} animateOpacity>
                      <Box p="20px" bg="gray.50" borderBottomRadius="md">
                        <Text fontSize="sm" color="gray.600" mb="4">
                          {row.original.description}
                        </Text>
                        <Flex direction="column" align="center">
                          <Input
                            placeholder="Enter tokens amount"
                            type="number"
                            value={tokensPerTask[row.original['task-id']] || ""}
                            onChange={(e) =>
                              handleTokensChange(row.original['task-id'], e.target.value)
                            }
                            width="300px"
                            mb="4"
                          />
                        </Flex>
                      </Box>
                    </Collapse>
                  </Td>
                </Tr>
              </React.Fragment>
            ))}
          </Tbody>
        </Table>
      </Box>

      {/* Modal for API and Training Details */}
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            Enter Training Details for Task {selectedTask ? selectedTask['task-id'] : 'Unknown Task'}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <FormControl mb="4">
              <FormLabel>FLOCK API KEY</FormLabel>
              <Input placeholder="Enter FLOCK API KEY" value={flockApiKey} onChange={(e) => setFlockApiKey(e.target.value)} />
            </FormControl>
            <FormControl mb="4">
              <FormLabel>Hugging Face Token</FormLabel>
              <Input placeholder="Enter Hugging Face Token" value={hfToken} onChange={(e) => setHfToken(e.target.value)} />
            </FormControl>
            <FormControl mb="4">
              <FormLabel>Hugging Face Username</FormLabel>
              <Input placeholder="Enter Hugging Face Username" value={hfUsername} onChange={(e) => setHfUsername(e.target.value)} />
            </FormControl>
          </ModalBody>
          <ModalFooter>
          <Button
            colorScheme="blue"
            mr={3}
            onClick={async () => {
              let output;
              try {
                const response = await axios.post("http://10.154.36.81:5000/execute", {
                  flockApiKey,
                  hfToken,
                  hfUsername,
                  taskId: selectedTask['task-id'],
                }).then((res) => {
                  output = res.error;
                  console.log("Response:", output);
                });
                console.log("Execution Successful:", response.data);
                alert("Training initiated successfully!");
                onClose(); // Close modal after successful request
              } catch (error) {
                console.error("Error in execution:", error.response?.data || error.message);
                // Show the output in an alert
                alert("Failed to initiate training. Please try again. Output: " + output);
                onClose();
              }
            }}
          >
            Submit
          </Button>
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Card>
  );
}
