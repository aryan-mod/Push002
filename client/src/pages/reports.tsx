import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon, Download, Search, Filter, MapPin, Clock, AlertTriangle } from "lucide-react";
import { format, parseISO, startOfDay, endOfDay } from "date-fns";
import Header from "@/components/layout/header";

/**
 * Reports Page Component
 * Displays historical alerts and reports with search/filter functionality
 * Includes data export capabilities and detailed incident analysis
 */
export default function Reports() {
  const [searchTerm, setSearchTerm] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [dateRange, setDateRange] = useState<{from?: Date; to?: Date}>({});
  
  // Fetch historical alerts and reports
  const { data: reports = [], isLoading, error } = useQuery<any[]>({
    queryKey: ["/api/v1/alerts", searchTerm, typeFilter, severityFilter, statusFilter, dateRange],
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Apply filters to reports
  const filteredReports = reports.filter((report: any) => {
    // Search filter
    if (searchTerm && !report.title.toLowerCase().includes(searchTerm.toLowerCase()) && 
        !report.message.toLowerCase().includes(searchTerm.toLowerCase())) {
      return false;
    }
    
    // Type filter
    if (typeFilter !== "all" && report.type !== typeFilter) {
      return false;
    }
    
    // Severity filter
    if (severityFilter !== "all" && report.severity !== severityFilter) {
      return false;
    }
    
    // Status filter
    if (statusFilter !== "all" && report.status !== statusFilter) {
      return false;
    }
    
    // Date range filter
    if (dateRange.from && dateRange.to) {
      const reportDate = parseISO(report.createdAt);
      if (reportDate < startOfDay(dateRange.from) || reportDate > endOfDay(dateRange.to)) {
        return false;
      }
    }
    
    return true;
  });

  // Export reports to CSV
  const handleExportReports = () => {
    const csvContent = generateCSV(filteredReports);
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `tourist-safety-reports-${format(new Date(), 'yyyy-MM-dd')}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  return (
    <>
      <Header 
        title="Safety Reports" 
        subtitle={`${filteredReports.length} incidents found across all locations`}
      />
      
      <div className="p-6">
        {/* Filter Controls */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Filter & Search Reports
            </CardTitle>
            <CardDescription>
              Find specific incidents or analyze patterns in tourist safety data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Search */}
              <div className="space-y-2">
                <Label>Search</Label>
                <div className="relative">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search incidents..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-8"
                    data-testid="input-search"
                  />
                </div>
              </div>

              {/* Type Filter */}
              <div className="space-y-2">
                <Label>Alert Type</Label>
                <Select value={typeFilter} onValueChange={setTypeFilter}>
                  <SelectTrigger data-testid="select-type-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="emergency">Emergency</SelectItem>
                    <SelectItem value="medical">Medical</SelectItem>
                    <SelectItem value="accident">Accident</SelectItem>
                    <SelectItem value="weather">Weather</SelectItem>
                    <SelectItem value="crime">Crime/Security</SelectItem>
                    <SelectItem value="natural_disaster">Natural Disaster</SelectItem>
                    <SelectItem value="wildlife">Wildlife</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Severity Filter */}
              <div className="space-y-2">
                <Label>Severity</Label>
                <Select value={severityFilter} onValueChange={setSeverityFilter}>
                  <SelectTrigger data-testid="select-severity-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Severities</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Status Filter */}
              <div className="space-y-2">
                <Label>Status</Label>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger data-testid="select-status-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Statuses</SelectItem>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="acknowledged">Acknowledged</SelectItem>
                    <SelectItem value="resolved">Resolved</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Date Range Picker */}
            <div className="mt-4 flex items-center gap-4">
              <div className="space-y-2">
                <Label>Date Range</Label>
                <div className="flex gap-2">
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant="outline" className="w-[240px] justify-start text-left font-normal">
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {dateRange.from ? (
                          dateRange.to ? (
                            <>
                              {format(dateRange.from, "LLL dd, y")} -{" "}
                              {format(dateRange.to, "LLL dd, y")}
                            </>
                          ) : (
                            format(dateRange.from, "LLL dd, y")
                          )
                        ) : (
                          <span>Pick a date range</span>
                        )}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        initialFocus
                        mode="range"
                        defaultMonth={dateRange.from}
                        selected={dateRange as any}
                        onSelect={(range) => setDateRange(range || {})}
                        numberOfMonths={2}
                      />
                    </PopoverContent>
                  </Popover>
                  
                  {(dateRange.from || dateRange.to) && (
                    <Button
                      variant="outline"
                      onClick={() => setDateRange({})}
                    >
                      Clear
                    </Button>
                  )}
                </div>
              </div>

              <div className="ml-auto">
                <Button 
                  onClick={handleExportReports}
                  variant="outline"
                  className="flex items-center gap-2"
                  data-testid="button-export-reports"
                >
                  <Download className="w-4 h-4" />
                  Export CSV
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Reports Table */}
        <Card>
          <CardHeader>
            <CardTitle>Incident Reports</CardTitle>
            <CardDescription>
              Comprehensive history of all safety alerts and emergency incidents
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading reports...</p>
                </div>
              </div>
            ) : error ? (
              <div className="text-center py-8">
                <AlertTriangle className="w-12 h-12 text-destructive mx-auto mb-4" />
                <p className="text-destructive">Failed to load reports</p>
              </div>
            ) : filteredReports.length === 0 ? (
              <div className="text-center py-8">
                <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No reports found matching your criteria</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date & Time</TableHead>
                      <TableHead>Incident Type</TableHead>
                      <TableHead>Severity</TableHead>
                      <TableHead>Location</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredReports.map((report: any) => (
                      <TableRow key={report.id} data-testid={`report-row-${report.id}`}>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Clock className="w-4 h-4 text-muted-foreground" />
                            <div>
                              <div className="font-medium">
                                {format(parseISO(report.createdAt), "MMM dd, yyyy")}
                              </div>
                              <div className="text-sm text-muted-foreground">
                                {format(parseISO(report.createdAt), "HH:mm")}
                              </div>
                            </div>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <Badge variant="outline" className="capitalize">
                            {report.type.replace('_', ' ')}
                          </Badge>
                        </TableCell>
                        
                        <TableCell>
                          <Badge 
                            variant={
                              report.severity === "critical" ? "destructive" :
                              report.severity === "high" ? "default" :
                              "secondary"
                            }
                          >
                            {report.severity.toUpperCase()}
                          </Badge>
                        </TableCell>
                        
                        <TableCell>
                          <div className="flex items-center gap-2 max-w-[200px]">
                            <MapPin className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                            <span className="truncate">
                              {report.site?.name || "Unknown Location"}
                            </span>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <div className="max-w-[300px]">
                            <p className="font-medium truncate">{report.title}</p>
                            <p className="text-sm text-muted-foreground truncate">
                              {report.message}
                            </p>
                          </div>
                        </TableCell>
                        
                        <TableCell>
                          <Badge 
                            variant={
                              report.status === "active" ? "destructive" :
                              report.status === "acknowledged" ? "default" :
                              "outline"
                            }
                          >
                            {report.status}
                          </Badge>
                        </TableCell>
                        
                        <TableCell>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => {
                              // TODO: Implement view details functionality
                              console.log("View report details:", report.id);
                            }}
                            data-testid={`button-view-details-${report.id}`}
                          >
                            View Details
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Summary Statistics */}
        {filteredReports.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-destructive">
                  {filteredReports.filter(r => r.severity === "critical").length}
                </div>
                <p className="text-sm text-muted-foreground">Critical Incidents</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-orange-600">
                  {filteredReports.filter(r => r.severity === "high").length}
                </div>
                <p className="text-sm text-muted-foreground">High Priority</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-green-600">
                  {filteredReports.filter(r => r.status === "resolved").length}
                </div>
                <p className="text-sm text-muted-foreground">Resolved</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold">
                  {Math.round((filteredReports.filter(r => r.status === "resolved").length / filteredReports.length) * 100)}%
                </div>
                <p className="text-sm text-muted-foreground">Resolution Rate</p>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </>
  );
}

/**
 * Generate CSV content from reports data
 */
function generateCSV(reports: any[]): string {
  const headers = [
    "Date",
    "Time", 
    "Type",
    "Severity",
    "Title",
    "Description",
    "Location",
    "Status",
    "Acknowledged By",
    "Acknowledged At"
  ];

  const rows = reports.map(report => [
    format(parseISO(report.createdAt), "yyyy-MM-dd"),
    format(parseISO(report.createdAt), "HH:mm:ss"),
    report.type,
    report.severity,
    report.title,
    report.message.replace(/"/g, '""'), // Escape quotes for CSV
    report.site?.name || "",
    report.status,
    report.acknowledgedBy || "",
    report.acknowledgedAt ? format(parseISO(report.acknowledgedAt), "yyyy-MM-dd HH:mm:ss") : ""
  ]);

  const csvContent = [headers, ...rows]
    .map(row => row.map(field => `"${field}"`).join(","))
    .join("\n");

  return csvContent;
}