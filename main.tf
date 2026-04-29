provider "aws" {
  region = "us-east-1"
}

# VPC Configuration
resource "aws_vpc" "NetID-25jpkj-vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  
  tags = {
    Name = "NetID-25jpkj-vpc"
  }
}

# Subnet Configuration
resource "aws_subnet" "NetID-25jpkj-subnet" {
  vpc_id                  = aws_vpc.NetID-25jpkj-vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true 
  availability_zone       = "us-east-1a"

  tags = {
    Name = "NetID-25jpkj-subnet"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "NetID-25jpkj-igw" {
  vpc_id = aws_vpc.NetID-25jpkj-vpc.id

  tags = {
    Name = "NetID-25jpkj-igw"
  }
}

# Route Table
resource "aws_route_table" "NetID-25jpkj-rt" {
  vpc_id = aws_vpc.NetID-25jpkj-vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.NetID-25jpkj-igw.id
  }

  tags = {
    Name = "NetID-25jpkj-rt"
  }
}

# Route Table Association
resource "aws_route_table_association" "NetID-25jpkj-assoc" {
  subnet_id      = aws_subnet.NetID-25jpkj-subnet.id
  route_table_id = aws_route_table.NetID-25jpkj-rt.id
}

#Security Group Configuration
resource "aws_security_group" "NetID-25jpkj-sg" {
  name        = "NetID-25jpkj-sg"
  description = "Allow SSH and Web Interface"
  vpc_id      = aws_vpc.NetID-25jpkj-vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "NetID-25jpkj-sg"
  }
}

# S3 Bucket for Storage
resource "aws_s3_bucket" "NetID-25jpkj-bucket" {
  bucket = "netid-25jpkj-cloud-storage-project"

  tags = {
    Name        = "NetID-25jpkj-bucket"
    Environment = "Project"
  }
}
