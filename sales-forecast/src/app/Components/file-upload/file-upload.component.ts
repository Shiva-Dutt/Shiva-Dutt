import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-file-upload',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css']
})
export class FileUploadComponent implements OnInit {

  selectedFile!: File;

  constructor(private http: HttpClient) { }

  ngOnInit(): void {
  }

  onFileSelected(event: any){
    this.selectedFile = <File>event.target.files[0];
  }

  onUpload(){
    //const fd = new FormData();
    //fd.append('csv', this.selectedFile, this.selectedFile.name)
    //this.http.post('Backend url',fd)
    //.subscribe(res => {
    //  console.log(res)
    //});
    
  }

}
