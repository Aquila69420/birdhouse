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
  Checkbox,
  Input,
  Button,
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
  const [selectedTask, setSelectedTask] = React.useState(null);
  const [tokens, setTokens] = React.useState("");
  const wallet_id = useSelector((state) => state.person.wallet_id);
  
  const textColor = useColorModeValue('secondaryGray.900', 'white');
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.100');

  // Toggle row expansion for expand icon only
  const toggleRowExpansion = (row) => {
    const rowId = row.id;
    setExpandedRows((prevState) => {
      const isExpanding = !prevState[rowId];
      
      // Clear the selected task if the row is collapsing
      if (!isExpanding) {
        setSelectedTask(null);
      } else {
        setSelectedTask(row.original); // Set the selected task if expanding
      }

      return {
        ...prevState,
        [rowId]: isExpanding,
      };
    });
  };

  // Handle task selection
  const handleCheckboxToggle = (row) => {
    const rowId = row.id;
    setExpandedRows((prevState) => ({
      ...prevState,
      [rowId]: !prevState[rowId],
    }));
    setSelectedTask(row.original); // Set the selected task for staking input
  };

  // Columns configuration
  const columns = [
    columnHelper.display({
      id: 'select',
      cell: (info) => (
        <Checkbox
          isChecked={selectedTask === info.row.original}
          onChange={() => handleCheckboxToggle(info.row)}
        />
      ),
    }),
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
    columnHelper.accessor('expand', {
      id: 'expand',
      header: () => null,
      cell: (info) => (
        <Icon
          as={expandedRows[info.row.id] ? MdExpandLess : MdExpandMore}
          w="20px"
          h="20px"
          cursor="pointer"
          onClick={(e) => {
            e.stopPropagation();
            toggleRowExpansion(info.row);
          }}
        />
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

  async function handleStake() {
    // Create an axios instance (optional)
    const axiosInstance = axios.create({
      baseURL: 'http://10.154.36.81:5000',
      headers: {
        'Content-Type': 'application/json',
        // Add authorization or other headers here if needed
      },
    });
    console.log("Stake Action:", { task: selectedTask, tokens });
    try {
      const response = await axiosInstance.post('/pay_tokens', {
        wallet_id,
        tokens,
      });
      setTokens(response.data);
      console.log("Token Update Successful:", response.data);
    } catch (error) {
      console.error("Error in token stake:", error.response?.data || error.message);
      alert("Failed to stake tokens. Please try again.");
    }
  };

  return (
    <Card flexDirection="column" w="100%" px="0px" overflowX="auto">
      <Flex px="25px" mb="8px" justifyContent="space-between" align="center">
        <Text color={textColor} fontSize="22px" fontWeight="700">
          Current Tasks
        </Text>
        <Menu />
      </Flex>
      <Box>
        <Table variant="simple" color="gray.500" mb="24px" mt="12px">
          <Thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <Tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <Th key={header.id} borderColor={borderColor}>
                    <Flex justifyContent="space-between" align="center">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                    </Flex>
                  </Th>
                ))}
              </Tr>
            ))}
          </Thead>
          <Tbody>
            {table.getRowModel().rows.map((row) => (
              <React.Fragment key={row.id}>
                <Tr
                  cursor="pointer"
                  onClick={() => toggleRowExpansion(row)}
                  _hover={{ bg: 'gray.100' }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <Td key={cell.id} borderColor="transparent">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </Td>
                  ))}
                </Tr>
                <Tr>
                  <Td colSpan={columns.length} p="0">
                    <Collapse in={expandedRows[row.id]} animateOpacity>
                      <Box p="20px" bg="gray.50" borderBottomRadius="md">
                        <Text fontSize="sm" color="gray.600" mb="4">
                          {row.original.description}
                        </Text>
                        <Flex direction="column" align="center">
                          <Input
                            placeholder="Enter tokens amount"
                            type="number"
                            value={tokens}
                            onChange={(e) => setTokens(e.target.value)}
                            width="300px"
                            mb="4"
                          />
                          <Button colorScheme="brandScheme" bg="brand.500" color="white">
                            Validate
                          </Button>
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
    </Card>
  );
}
