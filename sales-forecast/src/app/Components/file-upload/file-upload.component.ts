import { Component, OnInit } from '@angular/core';
import { FileUploadService } from 'src/app/Components/file-upload/file-upload.service';

@Component({
  selector: 'app-file-upload',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css']
})
export class FileUploadComponent implements OnInit {

  //shortLink: string = "";
  loading: boolean = false;
  file!: File;

  constructor(private fileUploadService: FileUploadService) { }

  ngOnInit(): void {
  }

  onChange(event: any){
    this.file = <File>event.target.files[0];
  }

  onUpload(){
    this.loading = !this.loading;
    console.log(this.file);
    this.fileUploadService.upload(this.file).subscribe(
      (event: any) =>{
        if(typeof (event) === 'object'){
          //this.shortLink = event.link;
          this.loading = false;
        }
      }
    );
  }

  routeToMl(){
    window.location.href = "http://127.0.0.1:5000/plot.png";
  }

}